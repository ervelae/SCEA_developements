from time import time
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances
from sklearn import preprocessing
from geopy.distance import geodesic
from pyproj import Geod


# --- Custom distance metric: directed half-ellipse ---
def half_ellipse_dist(src_point, target_points, *,
                     magnitudes, rotations,
                     b=1.0, rotation_unit="radians", include_sign=False,
                     dtype=np.float32):
    """
    Directed half-ellipse distance from one source to many targets.
    Calculates one "row" of a distance matrix.
    Each point has an associated magnitude and rotation (defining the half-ellipse parameters for that row).
    Returns (C,) float32.
    """
    x0, y0 = float(src_point[0]), float(src_point[1])
    Xc = target_points[:, 0].astype(dtype, copy=False)
    Yc = target_points[:, 1].astype(dtype, copy=False)

    ang = np.deg2rad(rotations) if rotation_unit == "degrees" else float(rotations)
    a = 1.0 + float(magnitudes)

    sin, cos = np.sin(ang), np.cos(ang)
    dX = Xc - x0
    dY = Yc - y0
    X_rot = cos * dX + sin * dY
    Y_rot = -sin * dX + cos * dY

    sign = np.sign(X_rot).astype(dtype) if include_sign else 1.0
    a_eff = np.where(X_rot >= 0.0, a, b).astype(dtype)

    return (sign * np.sqrt((X_rot / a_eff) ** 2 + (Y_rot / b) ** 2)).astype(dtype, copy=False)


# --- Custom pseudo-metric: directed half-ellipse on a geodesic ---
def geodesic_half_ellipse_dist(src_point, target_points, *,
                               magnitude, rotation, b,
                               rotation_unit="radians", include_sign=False,
                               distance_unit="km", dtype=np.float32):
    """
    Directed geodesic half-ellipse 'row' distance:
    - src_point, target_points: (lat, lon) in DEGREES on WGS84.
    - Builds local E/N using geodesic azimuth + distance.
    - Applies half-ellipse in the local frame aligned by `rotation`.

    Returns: (C,) dtype array of scaled distances.
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    x0, y0 = float(src_point[0]), float(src_point[1])  # lat, lon
    lat = target_points[:, 0].astype(float, copy=False)
    lon = target_points[:, 1].astype(float, copy=False)

    # Calculate distances and forward azimuth 
    # Forward azimuth: Initial angle from point a to point b (angle can change on an ellipsoid, thus "initial"). North is 0 deg, East is 90 deg, etc.
    az12_deg, _, dist_m = geod.inv(
        np.full_like(lon, y0),  # lon1 returns [x0, x0, ...., x0]
        np.full_like(lat, x0),  # lat1 returns [y0, y0, ...., y0]
        lon, # lon2 [x1, x2, ..., xn]
        lat, # lat2 [y1, y2, ..., yn]
    )

    # Choose units for the plane
    if distance_unit == "km":
        d = (dist_m / 1000.0).astype(float, copy=False)
        b_eff = float(b)
        a = 1.0 + float(magnitude)
    elif distance_unit == "m":
        d = dist_m.astype(float, copy=False)
        b_eff = float(b)
        a = 1.0 + float(magnitude)
    else:
        raise ValueError(f"Unknown distance_unit: {distance_unit}")

    # Turn the azimuth into East-wise distance (dE) and North-wise distance (dN)
    azr = np.deg2rad(az12_deg) # to radians
    dE = d * np.sin(azr) # distance to East
    dN = d * np.cos(azr) # distance to North

    # Rotate by wind angle
    ang = np.deg2rad(rotation) if rotation_unit == "degrees" else float(rotation)
    sin, cos = np.sin(ang), np.cos(ang) 
    X_rot = cos * dE + sin * dN
    Y_rot = -sin * dE + cos * dN

    # Apply half-ellipse formula in the rotated frame
    sign = np.sign(X_rot).astype(dtype) if include_sign else 1.0
    a_eff = np.where(X_rot >= 0.0, a, b_eff).astype(dtype, copy=False)

    return (sign * np.sqrt((X_rot / a_eff) ** 2 + (Y_rot / b_eff) ** 2)).astype(dtype, copy=False)



# This is the core row cache class used by the SCEA implementation below.
class RowDistanceCache:
    """
    Least recently used (LRU) cache of distance matrix *rows* (each row is a length-N float32 array; NaN means "unknown column").

    - Stores at most `max_rows` rows (evicts least-recently used when over capacity).
    - get_row_for_targets(i, cols): computes only missing columns for row i, writes them in,
      optionally cross-fills j->i if metric is symmetric AND row j is already cached.

    Instrumentation counters help tune `max_rows`.
    """

    def __init__(self, N, coords, metric,
                 *, max_rows=256, dtype=np.float32,
                 magnitude=None, rotation=None, rotation_unit="radians",
                 include_sign=False, b=1.0, radius=6371.0,
                 symmetric=None, cross_fill=False):
        self.N = int(N)
        self.coords = coords
        self.metric = metric
        self.dtype = np.dtype(dtype)

        # auto symmetry unless overridden
        self.symmetric = (metric in ("euclidean", "haversine")) if symmetric is None else bool(symmetric)
        self.cross_fill = bool(cross_fill)

        # half-ellipse parameters (row-dependent)
        self.magnitude = magnitude
        self.rotation = rotation
        self.rotation_unit = rotation_unit
        self.include_sign = include_sign
        self.b = b

        # haversine config
        self.radius = float(radius)  # Earth radius in km by default

        # LRU storage
        self.max_rows = int(max_rows)
        self.rows: "OrderedDict[int, np.ndarray]" = OrderedDict()

        # --- Stats / counters ---
        self.stats = {
            # capacity / memory
            "max_rows": self.max_rows,
            "approx_mem_rows": 0,            # number of rows currently cached
            "approx_mem_bytes": 0,           # approx memory footprint rows * N * 4 bytes
            # row lifecycle
            "rows_allocated": 0,             # how many distinct rows created
            "rows_evicted": 0,               # LRU evictions
            # requests & hits
            "row_requests": 0,               # total calls to get_row_for_targets
            "row_full_hits": 0,              # row existed AND all requested cols known
            "row_partial_hits": 0,           # row existed but had some missing cols
            "row_misses": 0,                 # row did not exist prior to this request
            # columns accounting
            "cols_requested": 0,             # total requested columns (sum of len(cols) across calls)
            "cols_missing": 0,               # how many were missing before compute
            "cols_computed": 0,              # how many we actually computed this call
            "cols_cross_filled": 0,          # how many reverse entries we wrote (if symmetric & cross_fill)
        }

    # ---------- internal helpers ----------
    def _update_memory_estimate(self):
        rows_now = len(self.rows)
        self.stats["approx_mem_rows"] = rows_now
        self.stats["approx_mem_bytes"] = rows_now * self.N * self.dtype.itemsize

    def _touch(self, i: int):
        """Mark row i as most-recently used (LRU bookkeeping)."""
        self.rows.move_to_end(i, last=True)

    def _ensure_row_alloc(self, i_global: int) -> np.ndarray:
        """
        Return cached row for i_global; allocate new row if needed.
        Evict LRU if at capacity.
        """
        row = self.rows.get(i_global)
        if row is not None:
            self._touch(i_global)
            return row

        # Evict if at capacity
        if self.max_rows > 0 and len(self.rows) >= self.max_rows:
            self.rows.popitem(last=False)
            self.stats["rows_evicted"] += 1

        # Allocate new NaN row; diagonal is 0
        row = np.full(self.N, np.nan, dtype=self.dtype)
        row[i_global] = 0.0
        self.rows[i_global] = row
        self._touch(i_global)

        self.stats["rows_allocated"] += 1
        self._update_memory_estimate()
        return row

    def _calc_distance(
        self,
        i_global: int,
        row: np.ndarray,
        cols: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
    ) -> np.ndarray:
        """
        Core metric dispatch that computes distances for (i_global -> cols),
        writes to `row[cols]`, and performs optional symmetric cross-fill.

        i_global: int index of the source point (row).
        row: the full row array for i_global (length N, with NaNs for unknown columns).
        cols: 1D array of target global indices for which distances are missing and need to be computed.
        P: (1, d) array of the source point's coordinates.
        Q: (C, d) array of the target points' coordinates (aligned with `cols`).

        Returns the computed values (aligned with `cols`).
        """
        metric = self.metric
        symmetric_metric = False

        if isinstance(metric, str):
            m = metric.lower()
            if m == "euclidean":
                vals = cdist(P, Q, "euclidean").astype(self.dtype, copy=False)[0]
                symmetric_metric = True

            elif m == "haversine":
                Pr = np.radians(P)
                Qr = np.radians(Q)
                vals = (haversine_distances(Pr, Qr)[0] * self.radius).astype(self.dtype, copy=False)
                symmetric_metric = True

            elif m == "half_ellipse":
                # directed: i -> j using *row* parameters
                mag_row = self.magnitude[i_global] if hasattr(self.magnitude, "__len__") else self.magnitude
                rot_row = self.rotation[i_global] if hasattr(self.rotation, "__len__") else self.rotation
                vals = half_ellipse_dist(
                    src_point=P[0], target_points=Q,
                    magnitudes=mag_row, rotations=rot_row,
                    b=self.b, rotation_unit=self.rotation_unit, include_sign=self.include_sign,
                    dtype=self.dtype,
                ).astype(self.dtype, copy=False)
                symmetric_metric = False

            elif m == "geodesic":
                # Expect (lat, lon) in degrees
                src_lat = float(P[0, 0])
                src_lon = float(P[0, 1])
                lat = Q[:, 0].astype(float, copy=False)
                lon = Q[:, 1].astype(float, copy=False)
                geod = Geod(ellps="WGS84")
                _, _, dist_m = geod.inv(
                    np.full_like(lon, src_lon),
                    np.full_like(lat, src_lat),
                    lon, lat
                )
                vals = (dist_m / 1000.0).astype(self.dtype, copy=False)  # km
                symmetric_metric = True

            elif m == "geodesic_half_ellipse":
                mag_row = self.magnitude[i_global] if hasattr(self.magnitude, "__len__") else self.magnitude # if only a scalar, then use that
                rot_row = self.rotation[i_global] if hasattr(self.rotation, "__len__") else self.rotation # if only a scalar, then use that

                vals = geodesic_half_ellipse_dist(
                    src_point=P[0], target_points=Q,
                    magnitude=mag_row, rotation=rot_row,
                    b=self.b, rotation_unit=self.rotation_unit,
                    include_sign=self.include_sign,
                    distance_unit="km",
                    dtype=self.dtype
                )
                symmetric_metric = False


            else:
                raise ValueError(f"Unknown metric: {self.metric}")

        else:
            # Callable metric: metric(P, Q) -> 1D array-like
            vals = np.asarray(metric(P, Q), dtype=self.dtype).reshape(-1)
            # Cross-fill governed by self.symmetric for callables.
            symmetric_metric = bool(self.symmetric)

        # Write row
        row[cols] = vals

        # Optional symmetric cross fill (only if both metric and settings allow)
        if symmetric_metric and self.symmetric and self.cross_fill:
            cf = 0
            for j, v in zip(cols, vals):
                rj = self.rows.get(int(j))
                if rj is not None:
                    rj[i_global] = v
                    cf += 1
            self.stats["cols_cross_filled"] += cf

        return vals

    # ---------- main API ----------
    def get_row_for_targets(self, i_global, target_globals):
        """
        Ensure row distances i_global -> target_globals exist; compute missing only.
        Returns distances in the order of target_globals as a 1D float32 array.
        """
        i_global = int(i_global)
        target_globals = np.asarray(target_globals, dtype=np.int64)

        # Stats: request-level accounting
        self.stats["row_requests"] += 1
        self.stats["cols_requested"] += int(target_globals.size)

        # Did the row exist prior to this request?
        row_existed_before = (i_global in self.rows)

        row = self._ensure_row_alloc(i_global)  # allocates if needed
        missing_mask = np.isnan(row[target_globals])
        num_missing = int(missing_mask.sum())
        self.stats["cols_missing"] += num_missing

        # Classify hit/miss
        if row_existed_before:
            if num_missing == 0:
                self.stats["row_full_hits"] += 1
            else:
                self.stats["row_partial_hits"] += 1
        else:
            # brand new row (we count as "miss" regardless of missing count)
            self.stats["row_misses"] += 1

        # Nothing to compute
        if num_missing == 0:
            return row[target_globals]

        # Compute only missing columns now
        cols = target_globals[missing_mask]
        P = self.coords[i_global:i_global+1]  # (1, d)
        Q = self.coords[cols]                 # (C, d)

        # Centralized metric dispatch + cross-fill
        _ = self._calc_distance(i_global, row, cols, P, Q)

        self._touch(i_global)
        self.stats["cols_computed"] += num_missing
        return row[target_globals]

    # ---------- stats helpers ----------
    def reset_stats(self):
        cap = self.max_rows  # keep capacity
        self.stats = {
            "max_rows": cap,
            "approx_mem_rows": len(self.rows),
            "approx_mem_bytes": len(self.rows) * self.N * self.dtype.itemsize,
            "rows_allocated": 0,
            "rows_evicted": 0,
            "row_requests": 0,
            "row_full_hits": 0,
            "row_partial_hits": 0,
            "row_misses": 0,
            "cols_requested": 0,
            "cols_missing": 0,
            "cols_computed": 0,
            "cols_cross_filled": 0,
        }

    def stats_dict(self):
        # Also compute derived rates here
        s = dict(self.stats)  # shallow copy
        req = max(1, s["row_requests"])
        s["row_full_hit_rate"] = s["row_full_hits"] / req
        s["row_any_hit_rate"] = (s["row_full_hits"] + s["row_partial_hits"]) / req
        s["cols_fill_rate"] = (s["cols_computed"] / max(1, s["cols_requested"]))
        return s

    def stats_summary(self) -> str:
        s = self.stats_dict()
        mem_mb = s["approx_mem_bytes"] / (1024**2)
        return (
            f"RowDistanceCache stats:\n"
            f"  capacity (rows):        {s['max_rows']}\n"
            f"  resident rows:          {s['approx_mem_rows']} (~{mem_mb:.1f} MB)\n"
            f"  rows allocated:         {s['rows_allocated']}\n"
            f"  rows evicted (LRU):     {s['rows_evicted']}\n"
            f"  row requests:           {s['row_requests']}\n"
            f"    full hits:            {s['row_full_hits']}  "
            f"(rate={s['row_full_hit_rate']:.2%})\n"
            f"    partial hits:         {s['row_partial_hits']}\n"
            f"    misses:               {s['row_misses']}\n"
            f"  cols requested:         {s['cols_requested']}\n"
            f"  cols missing:           {s['cols_missing']}\n"
            f"  cols computed:          {s['cols_computed']}  "
            f"(fill-rate={s['cols_fill_rate']:.2%})\n"
            f"  cols cross-filled:      {s['cols_cross_filled']}\n"
            f"  metric:                 {self.metric}  "
            f"(symmetric={self.symmetric}, cross_fill={'on' if self.cross_fill else 'off'})"
        )







# Used by the main SCEA implementation
def find_one_cluster(points, values, get_row,
                             radius_func="default",
                             growth_limit=2,
                             max_pts_start_radius=7):
    """
    Cluster growth using a row-getter: get_row(i_local) -> 1D distances to all local points.
    Computes only the rows that are needed (radiating points).
    """
    if radius_func == "default":
        radius_func = lambda x: min(1 + x - growth_limit, 2)

    values = (values - values.mean()) / (values.std() + 1e-12)

    N = len(points)
    in_cluster = np.zeros(N, dtype=bool)
    radiating = np.zeros(N, dtype=bool)

    i0 = np.argmax(values)
    in_cluster[i0] = True
    radiating[i0] = True

    while True:
        remaining = ~in_cluster
        if not remaining.any():
            break

        new_pts = np.zeros(N, dtype=bool)

        for idx in np.where(radiating)[0]:
            radius_coef = radius_func(values[idx])
            if radius_coef <= 1:
                radiating[idx] = False
                continue

            row = get_row(idx)  # distances to all local points in order

            # nearest outsider distance (start radius)
            d0 = np.min(row[remaining])

            # kth neighbor threshold (unfiltered row)
            k = min(N - 1, max_pts_start_radius)
            d_k = np.partition(row, k)[k]

            if d0 >= d_k:
                radiating[idx] = False
                continue

            r = d0 * radius_coef
            cand = (row < r) & remaining
            if cand.any():
                new_pts |= cand
            else:
                radiating[idx] = False

        if not new_pts.any():
            break

        in_cluster |= new_pts
        radiating |= new_pts

    return in_cluster


def reinsert_nans_as_to_output(clusters, nan_mask):
    if nan_mask.any():
        full_clusters = np.full(len(nan_mask), np.nan, dtype=float)
        full_clusters[~nan_mask] = clusters
        return full_clusters
    else:
        return clusters


# TODO
def print_cluster_stats(clusters, values):
    pass


# ----------------------------- SCEA main implementation ----------------------------- #
def scea(
    coords,
    values,
    *,
    growth_limit=2,
    detection_limit=3.5,
    max_pts_start_radius=5,
    local_box_size=0,
    metric="euclidean",
    radius_func="default",
    n_clusters="auto",
    point_value_threshold="stds_from_median",
    distance_matrix_kwargs=None,
    row_cache_max_rows=256,
    symmetric_cross_fill=False,
    verbose=True,
):
    """
    SCEA using a cache of distance matrix rows for efficient calculations.
    - Maintains a boolean 'not_clustered' mask.
    - Per seed: builds local set, binds a row-getter to that set, grows cluster.
    - Stores rows only for radiating points (and only columns used so far).
    """

    # Turn to numpy arrays
    coords = np.asarray(coords)
    values = np.asarray(values)

    shape = None
    # Reshape values to 1D if needed
    if values.ndim > 1:
        shape = values.shape
        values = values.flatten()
    # Reshape coords to (N, 2) if needed
    if (coords.ndim == 2 and coords.shape[1] != 2 and coords.shape[0] == 2): # if (2, N)
        coords = coords.T
    elif (coords.ndim > 2 and coords.shape[0] == 2): # if (2, ..., ...)
        coords = coords.reshape(2, -1).T
    #elif (coords.ndim > 2 and coords.shape[-1] == 2): # if (..., ..., 2)
    #    coords = coords.reshape(-1, 2)    

    # Remove NaN points
    nan_mask = np.isnan(values)
    if nan_mask.any():
        coords = coords[~nan_mask]
        values = values[~nan_mask]
        if verbose >= 1:
            print(f"[Init] Warning: {nan_mask.sum()} points with NaN values found. They will be ignored.")

    # Initialize
    N = len(values)
    clusters = np.zeros(N, dtype=int)
    cluster_id = 1

    # === Stopping threshold setup ===
    if n_clusters == "auto":
        if point_value_threshold == "stds_from_mean":
            pv_std = preprocessing.StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
            pv_std[pv_std < detection_limit] = np.inf
            if np.isinf(pv_std).all():
                if verbose >= 1:
                    print(f"[-] No clusters found. All points are too close to the mean. Lower detection_limit={detection_limit}?")
                # Reinsert NaN points as cluster 0 (unclustered)
                if nan_mask.any():
                    clusters = reinsert_nans_as_to_output(clusters, nan_mask)
                return clusters
            point_value_threshold = values[np.argmin(pv_std)]
        elif point_value_threshold == "stds_from_median":
            median, std = np.median(values), np.std(values)
            pv_std = (values - median) / (std + 1e-12)
            pv_std[pv_std < detection_limit] = np.inf
            if np.isinf(pv_std).all():
                if verbose >= 1:
                    print(f"[-] No clusters found. All points are too close to the median. Lower detection_limit={detection_limit}?")
                # Reinsert NaN points as cluster 0 (unclustered)
                if nan_mask.any():
                    clusters = reinsert_nans_as_to_output(clusters, nan_mask)
                return clusters
            point_value_threshold = values[np.argmin(pv_std)]
            if verbose >= 1:
                print(f"[Init] Auto threshold set: {point_value_threshold:.7f}. (detection_limit=){detection_limit} standard deviations from median.  (median={median:.7f}, std={std:.7f})")
    else:
        point_value_threshold = -np.inf


    if verbose >= 1:
        print(f"[Start] SCEA: metric={metric}, growth_limit={growth_limit}, detection_limit={detection_limit}, local_box_size={local_box_size}, max_pts_start_radius={max_pts_start_radius}, n_clusters={n_clusters}")
        # Time the execution
        start_time = time()


    # --- Instantiate a lightweight row cache (global over the whole run) ---
    dm_kwargs = distance_matrix_kwargs or {}
    row_cache = RowDistanceCache(
        N=N,
        coords=coords,
        metric=metric,
        max_rows=row_cache_max_rows,
        dtype=np.float32,
        magnitude=dm_kwargs.get("magnitude"),
        rotation=dm_kwargs.get("rotation"),
        rotation_unit=dm_kwargs.get("rotation_unit", "radians"),
        include_sign=dm_kwargs.get("include_sign", False),
        b=dm_kwargs.get("b", 1.0),
        symmetric=None,  # auto: True for euclidean/haversine, False for half_ellipse
        cross_fill=symmetric_cross_fill,
    )

    not_clustered = np.ones(N, dtype=bool)

    while not_clustered.any():
        active_idx = np.where(not_clustered)[0]
        active_coords = coords[active_idx]
        active_values = values[active_idx]

        # stopping
        if active_values.max() <= point_value_threshold:
            if verbose >= 1:
                print(f"\n[Stop] Threshold reached: max active {active_values.max():.7f} <= {point_value_threshold:.7f}")
            break

        # choose seed in active set
        i0_local = np.argmax(active_values)
        center = active_coords[i0_local]

        # local box in active space
        if local_box_size == 0:
            local_idx_active = np.arange(len(active_idx), dtype=np.int64)
        else:
            half = local_box_size / 2.0
            mask = (
                (np.abs(active_coords[:, 0] - center[0]) <= half) &
                (np.abs(active_coords[:, 1] - center[1]) <= half)
            )
            local_idx_active = np.where(mask)[0].astype(np.int64)

        if local_idx_active.size == 0:
            # shouldn't happen; just drop the seed
            not_clustered[active_idx[i0_local]] = False
            continue

        local_globals = active_idx[local_idx_active]
        local_coords = coords[local_globals]
        local_values = values[local_globals]

        # Bind a row getter to this local set:
        # get_row(i_local) -> distances from global point local_globals[i_local] to all local_globals (aligned order)
        def get_row(i_local: int):
            i_global = int(local_globals[i_local])
            return row_cache.get_row_for_targets(i_global=i_global, target_globals=local_globals)

        # Grow cluster in this local set
        selected_mask = find_one_cluster(
            points=local_coords,
            values=local_values,
            get_row=get_row,
            radius_func=radius_func,
            growth_limit=growth_limit,
            max_pts_start_radius=max_pts_start_radius
        )

        chosen_globals = local_globals[selected_mask]
        clusters[chosen_globals] = cluster_id
        not_clustered[chosen_globals] = False

        if verbose >= 2:
            print(f"[{cluster_id}] Cluster formed. Size={selected_mask.sum()} | Points left: {not_clustered.sum()} | "
                  f"Seed coords & value: {local_coords[np.argmax(local_values)]}, {active_values.max():.6f} | ")
        if verbose == 1:
            print(f"\r[{cluster_id - 1}] Cluster formed. | current max value: {active_values.max():.7f} | stopping threshold: {point_value_threshold:.7f}", end='', flush=True)


        cluster_id += 1

        if isinstance(n_clusters, int) and cluster_id > n_clusters:
            if verbose >= 1:
                print(f"\n[Stop] Reached n_clusters={n_clusters}.")
            break

    if verbose >= 2:
        print("[Stats]" + row_cache.stats_summary())
        print_cluster_stats(clusters, values)
    if verbose >= 1:
        end_time = time()
        print(f"[Done] Total clusters found: {cluster_id - 1}, Time taken: {end_time - start_time:.2f} seconds")

    # Reinsert NaN points as cluster 0 (unclustered)
    if nan_mask.any():
        clusters = reinsert_nans_as_to_output(clusters, nan_mask)

    # Reshape clusters to match original values shape
    if shape:
        clusters = clusters.reshape(shape)

    return clusters