import os
from time import time
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances
from sklearn import preprocessing
from geopy.distance import geodesic
from pyproj import Geod
import warnings
import colorcet as cc
import xarray as xr
from numba import njit, prange
import bisect
from shapely.geometry import Point
import geopandas as gpd
from haversine import haversine_vector, Unit
from scipy.spatial import KDTree
import pandas as pd


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



















# ======================= SCEA main implementation ======================== #
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
    tqdm=None,
):
    """
    SCEA using a cache of distance matrix rows for efficient calculations.
    - Maintains a boolean 'not_clustered' mask.
    - Per seed: builds local set, binds a row-getter to that set, grows cluster.
    - Stores rows only for radiating points (and only columns used so far).
    """

    if verbose >= 1:
        print(f"[Init] Starting SCEA with {len(values)} points.")
    if tqdm is not None:
        tqdm.set_description(f"   [Init] Starting SCEA with {len(values)} points.")

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
        if tqdm is not None:
            tqdm.set_description(f"  [{cluster_id - 1}] Cluster formed. | current max value: {active_values.max():.7f} | stopping threshold: {point_value_threshold:.7f}")

        cluster_id += 1

        if isinstance(n_clusters, int) and cluster_id > n_clusters:
            if verbose >= 1:
                print(f"\n[Stop] Reached n_clusters={n_clusters}.")
            break

    end_time = time()
    if verbose >= 2:
        print("[Stats]" + row_cache.stats_summary())
        print_cluster_stats(clusters, values)
    if verbose >= 1:
        print(f"[Done] Total clusters found: {cluster_id - 1}, Time taken: {end_time - start_time:.2f} seconds")
    if tqdm is not None:
        tqdm.set_description(f"  [Done] Total clusters found: {cluster_id - 1}, Time taken: {end_time - start_time:.2f} seconds")

    # Reinsert NaN points as cluster 0 (unclustered)
    if nan_mask.any():
        clusters = reinsert_nans_as_to_output(clusters, nan_mask)

    # Reshape clusters to match original values shape
    if shape:
        clusters = clusters.reshape(shape)

    return clusters



























# ====================================================================================
# SCEA helper functions and utils (preprocessing, etc.)
# ====================================================================================


from scipy.ndimage import gaussian_filter, generic_filter, uniform_filter
import xarray as xr



def wind_to_angle_and_magnitude(wind_vector):
    """
    Converts a wind vector or array of wind vectors to angle(s) and magnitude(ies).
    
    Parameters:
    wind_vector : np.ndarray
        Wind vector(s) with the last dimension being 2 (for [wx, wy]).
        Can be 1D [wx, wy], 2D (n, 2), 3D (n, m, 2), or higher dimensions.
    
    Returns:
    tuple
        (wind_angles, wind_magnitudes) with the same shape as the input,
        except the last dimension is removed (since it contained [wx, wy]).
        wind_angle(s) are in radians.
    
    Example:
        Single vector [1, 1] -> scalar, scalar
        Multiple vectors (3, 2) -> (3,), (3,)
        Grid of vectors (n, m, 2) -> (n, m), (n, m)
    """
    wind_vector = np.asarray(wind_vector)
    
    if wind_vector.shape[-1] != 2:
        raise ValueError(f"Last dimension must be 2 (for [wx, wy]), got shape {wind_vector.shape}")
    
    # Avoid -0.0 issues
    wind_vector = wind_vector.copy()
    wind_vector[wind_vector == 0] = 0
    
    # Extract wx and wy using the last dimension
    wx = wind_vector[..., 0]
    wy = wind_vector[..., 1]
    
    wind_magnitude = np.sqrt(wx ** 2 + wy ** 2)
    wind_angle = np.arctan2(wy, wx)
    
    return wind_angle, wind_magnitude



def match_era5_wind_to_tropomi(tropomi_data, era5_data, era5_variables=['wind_magnitude_100m', 'wind_angle_100m', 'wind_magnitude_10m', 'wind_angle_10m']):
    """
    Matches ERA5 wind data to TROPOMI data based on spatial and temporal coordinates.
    """

    # Convert TROPOMI time to np.datetime64 if it's not already, suppressing timezone warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="no explicit representation of timezones available for np.datetime64",
            category=UserWarning,
        )
        tropomi_data["time_utc"] = xr.apply_ufunc(
            lambda x: np.datetime64(x, "ns"),
            tropomi_data["time_utc"],
            vectorize=True,
            dask="allowed",
            output_dtypes=[np.dtype("datetime64[ns]")],
        )

    wind_data = {}

    # Select the nearest ERA5 grid point for each TROPOMI point in space and time, and extract the specified wind variables.
    for var in era5_variables:
        wind_data[var] = era5_data[var].sel(
            latitude=tropomi_data.latitude,
            longitude=tropomi_data.longitude,
            time=tropomi_data.time_utc.broadcast_like(tropomi_data["longitude"][0]),
            method="nearest",
        )
    
    return wind_data



def scale_wind_magnitude_for_distance_matrix(wind_magnitudes, method="linear", parameter=None):
    if method == "linear":
        if parameter is None:
            parameter = 0.03  # default linear scaling factor
        return parameter * wind_magnitudes
    elif method == "sqrt":
        if parameter is None:
            parameter = 0.2  # default sqrt scaling factor
        return parameter * np.sqrt(wind_magnitudes)
    elif method == "log":
        if parameter is None:
            parameter = 0.3  # default log scaling factor
        return parameter * np.log(1 + wind_magnitudes)
    else:
        raise ValueError("Unknown scaling method")
    




# ==== Data preprocessing utilities ====

# Crop TROPOMI dataset to given bbox
def crop_to_bbox(
    ds, 
    bbox: list[float] = [-10.0, 35.0, 30.0, 70.0],  # [min_lon, min_lat, max_lon, max_lat]
    save_path: str = None,
    variable_names = None,
    verbose: bool = True
) -> xr.Dataset:
    """
    Crop xarray Dataset to given bbox.
    """

    if type(ds) is not xr.Dataset:
        ds = xr.open_dataset(ds, group="PRODUCT", mask_and_scale=True)

    if verbose:
        print("[+] Cropping data files.")

    if variable_names is None:
        variable_names = {"lon": "longitude", "lat": "latitude", "value": "nitrogendioxide_tropospheric_column"}

    if ds[variable_names["value"]].ndim == 3:
        lon = ds[variable_names["lon"]][0].data
        lat = ds[variable_names["lat"]][0].data
    else:
        lon = ds[variable_names["lon"]].data
        lat = ds[variable_names["lat"]].data


    # Mask for points inside bbox
    in_bbox = (lon >= bbox[0]) & (lon <= bbox[2]) & \
                (lat >= bbox[1]) & (lat <= bbox[3])

    ds_crop = None

    if np.any(in_bbox):

        # Find valid rows/cols (at least one point in bbox)
        valid_rows = np.any(in_bbox, axis=1)
        valid_cols = np.any(in_bbox, axis=0)

        in_bbox = in_bbox[valid_rows, :][:, valid_cols]

        ds_crop = ds.isel(scanline=valid_rows, ground_pixel=valid_cols)

        for vars in ds_crop.variables:
            if (("scanline" in ds_crop[vars].coords) 
                and ("ground_pixel" in ds_crop[vars].coords) 
                and (ds_crop[vars].shape == in_bbox.shape) 
                and (vars not in ["time", "latitude", "longitude", "time_utc"])
                ):
                #print("[+] Found 'scanline' and 'ground_pixel' coordinates in", vars)
                # Turn values outside bbox to NaN
                ds_crop[vars] = ds_crop[vars].where(in_bbox)

    else:
        if verbose:
            print(f"[-] No data in bbox for file")


    if save_path:
        ds_crop.to_netcdf(save_path, group="PRODUCT")
        if verbose:
            print(f"[+] Saved cropped dataset to {save_path}")

    return ds_crop


def quality_filter(ds, qa_var="qa_value", value_var="nitrogendioxide_tropospheric_column", threshold=0.75, verbose=True):
    """
    Filter xarray Dataset based on quality assurance variable.
    """
    if qa_var not in ds:
        print(f"[-] QA variable '{qa_var}' not found in dataset. Skipping quality filtering.")
        return ds

    if verbose: print(f"[+] Applying quality filter: {qa_var} >= {threshold}")
    mask = ds[qa_var] >= threshold
    ds[value_var] = ds[value_var].where(mask)

    if verbose:
        total_points = ds[qa_var].size
        valid_points = mask.sum().item()
        print(f"[+] Quality filter applied. Valid points: {valid_points}/{total_points} ({(valid_points/total_points)*100:.2f}%)")
    return ds


def discard_files_smaller_than(data, variable_name, min_valid_points=20, verbose=True):
    """
    Discard xarray Datasets that have fewer than min_valid_points valid (non-NaN) data points in the main variable of interest.
    """
    #if verbose: print(f"[+] Discarding files with fewer than {min_valid_points} valid points.")
    valid_points = np.sum(np.isfinite(data[variable_name].values))
    if valid_points < min_valid_points:
        if verbose: print(f"[-] Discarding file with only {valid_points} valid points.")
        return None
    else:
        if verbose: print(f"[+] Keeping file with {valid_points} valid points.")
        return data





# ==== NaN-aware filters for smoothing and local standardization ====

def nan_gaussian_filter(data, sigma=1.0, mode="nearest"):
    """
    Applies a Gaussian filter to data that may contain NaNs, treating NaNs as missing values and not letting them contribute to the smoothing of valid data. 
    The output will have NaN in any position where the original data had NaN, and valid smoothed values elsewhere.

    """
    mask = np.isfinite(data).astype(float)
    
    # Fill NaNs with zeros for convolution; the mask will correct for this later.
    data_filled = np.where(np.isfinite(data), data, 0.0)

    # Apply Gaussian filter to both the filled data
    ## The result will be biased low near NaNs(=0.0), but we will correct for that using the weights from the mask.
    smooth = gaussian_filter(data_filled, sigma=sigma, mode=mode)

    # Apply Gaussian filter to the mask to get the effective weights (sum of kernel over valid points)
    ## If a point has NaN neighbors, the weight will be less than 1, and we can use this to correct the smoothed value.
    weight = gaussian_filter(mask, sigma=sigma, mode=mode) 

    # Avoid division by zero: where weight is zero, we set the result to NaN (since it means there were no valid points contributing to the smooth value).
    result = np.divide(smooth, weight, where=(weight > 0), out=np.full_like(smooth, np.nan))

    # Finally, we also want to ensure that any point that was originally NaN remains NaN in the output, even if the smoothed value is non-NaN due to neighboring values.
    return np.where(mask > 0, result, np.nan)


def local_standardization(da, window=11, eps=1e-12):
    """
    Local standardization using a sliding window.
    """

    # Windows must be odd to have a center pixel
    if isinstance(window, int):
        if window % 2 == 0:
            raise ValueError("Window size must be odd.")
    else:
        if window[0] % 2 == 0 or window[1] % 2 == 0:
            raise ValueError("Window sizes must be odd.")

    # accept either DataArray or ndarray
    data = da.values[0] if isinstance(da, xr.DataArray) else np.asarray(da)

    # Create a mask for finite values and fill NaNs with zeros for convolution
    ## The mask will be used to correct the sums and counts to get the true local mean and std, while the filled data allows us to use uniform_filter without NaN issues.
    mask = np.isfinite(data).astype(float)
    data_filled = np.where(np.isfinite(data), data, 0.0)

    # Compute local sums and sums of squares using uniform_filter, which gives us the total sum in each window. We will divide by the effective count of valid points (from the mask) to get the mean and variance.
    sum_ = uniform_filter(data_filled, window) * (window ** 2)
    sumsq = uniform_filter(data_filled**2, window) * (window ** 2)

    # The effective number of valid points in each window (the sum of the mask) is needed to compute the mean and variance correctly, since uniform_filter will treat NaNs as zeros.
    w = uniform_filter(mask, window) * (window ** 2)

    # Compute local mean and variance, correcting for the number of valid points. Where w is zero (no valid points), we set mean and std to NaN.
    mean = np.where(w > 0, sum_ / (w+eps), np.nan)

    # Variance is E[X^2] - (E[X])^2, where E[X^2] is sumsq / w and E[X] is mean. We also need to ensure that we don't get negative variance due to numerical issues, hence the np.maximum with 0.
    var = np.where(w > 0, sumsq / (w+eps) - mean**2, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))

    standardized = (data - mean) / (std + eps)

    # zero values back to NaN
    standardized = np.where(mask > 0, standardized, np.nan)
    
    return standardized


def nanmedian_filter(data, window):
    """
    NaN-aware median filter matching:
        scipy.ndimage.generic_filter(..., np.nanmedian, mode="constant", cval=np.nan)

    Supports:
        - numpy.ndarray
        - xarray.DataArray (uses data.values[0])
    """

    # Handle xarray input
    if isinstance(data, xr.DataArray):
        arr = data.values[0]
    else:
        arr = np.asarray(data)

    H, W = arr.shape

    # Window size handling
    if isinstance(window, int):
        wy = wx = window
    else:
        wy, wx = window

    pad_y = wy // 2
    pad_x = wx // 2

    # Pad with NaNs
    padded = np.pad(
        arr,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=np.nan,
    )

    # Initialize output with NaNs
    out = np.full((H, W), np.nan, dtype=float)

    # Median from sorted list
    def median_of_sorted(lst):
        n = len(lst)
        if n == 0:
            return np.nan
        m = n // 2
        if n % 2:
            return lst[m]
        return 0.5 * (lst[m - 1] + lst[m])

    # ==== MAIN LOOP ====

    for y in range(H):

        # === INITIAL WINDOW (x = 0) ===
        block = padded[y:y+wy, 0:wx]
        finite = block[np.isfinite(block)]
        values = list(finite)
        values.sort()

        out[y, 0] = median_of_sorted(values)

        # === SLIDE HORIZONTALLY ===
        for x in range(1, W):

            # --- Remove outgoing column ---
            col_out = padded[y:y+wy, x-1]
            finite_out = col_out[np.isfinite(col_out)]
            for v in finite_out:
                idx = bisect.bisect_left(values, v)
                values.pop(idx)

            # --- Add incoming column ---
            col_in = padded[y:y+wy, x-1+wx]
            finite_in = col_in[np.isfinite(col_in)]
            for v in finite_in:
                bisect.insort(values, v)

            out[y, x] = median_of_sorted(values)

    # Preserve original NaNs
    mask = np.isfinite(arr)
    return np.where(mask, out, np.nan)



@njit
def _binary_search_numba(arr, size, val):
    """Return insertion index for val in sorted arr[:size]."""
    lo = 0
    hi = size
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit
def _compute_median_numba(buf, size):
    if size == 0:
        return np.nan
    mid = size // 2
    if size % 2 == 1:
        return buf[mid]
    else:
        return 0.5 * (buf[mid - 1] + buf[mid])

@njit(parallel=True)
def _nanmedian_filter_kernel_numba(padded, H, W, wy, wx):
    out = np.empty((H, W), dtype=np.float64)
    maxk = wy * wx

    # ---------- PARALLEL ROW PROCESSING ----------
    for y in prange(H):

        # Thread-private working buffer
        buf = np.empty(maxk, dtype=np.float64)
        size = 0

        # ---- INITIAL WINDOW (x=0) ----
        size = 0
        for yy in range(y, y + wy):
            row = padded[yy, 0:wx]
            for v in row:
                if not np.isnan(v):
                    idx = _binary_search_numba(buf, size, v)
                    # shift right
                    for k in range(size, idx, -1):
                        buf[k] = buf[k - 1]
                    buf[idx] = v
                    size += 1

        out[y, 0] = _compute_median_numba(buf, size)

        # ---- SLIDE HORIZONTALLY ----
        for x in range(1, W):

            # --- remove outgoing column ---
            for yy in range(y, y + wy):
                v = padded[yy, x - 1]
                if not np.isnan(v):
                    idx = _binary_search_numba(buf, size, v)
                    # remove at idx
                    for k in range(idx, size - 1):
                        buf[k] = buf[k + 1]
                    size -= 1

            # --- add incoming column ---
            for yy in range(y, y + wy):
                v = padded[yy, x - 1 + wx]
                if not np.isnan(v):
                    idx = _binary_search_numba(buf, size, v)
                    # shift to make room
                    for k in range(size, idx, -1):
                        buf[k] = buf[k - 1]
                    buf[idx] = v
                    size += 1

            out[y, x] = _compute_median_numba(buf, size)

    return out


def nanmedian_filter_numba(data, window):
    """
    NaN-aware median filter using a Numba-accelerated sliding window.
    Matches generic_filter(np.nanmedian, mode="constant", cval=np.nan).
    """

    # Handle xarray input
    if isinstance(data, xr.DataArray):
        arr = data.values[0]
    else:
        arr = np.asarray(data)

    H, W = arr.shape

    if isinstance(window, int):
        wy = wx = window
    else:
        wy, wx = window

    pad_y = wy // 2
    pad_x = wx // 2

    padded = np.pad(arr,
                    ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant",
                    constant_values=np.nan)

    out = _nanmedian_filter_kernel_numba(padded, H, W, wy, wx)

    # Preserve original NaNs
    mask = np.isfinite(arr)
    return np.where(mask, out, np.nan)



def local_standardization_m_mad_numba(da, window=11, eps=0, scaling_factor=1.4826):
    return local_standardization_m_mad(da, window=window, eps=eps, scaling_factor=scaling_factor, use_numba=True)


def local_standardization_m_mad(da, window=11, eps=0, scaling_factor=1.4826, use_numba=True):
    """
    Modified MAD standardization
    Local standardization using local median and local modified MAD (median of absolute deviations from local median).
    """

    # accept either DataArray or ndarray
    data = da.values[0] if isinstance(da, xr.DataArray) else np.asarray(da)

    # keep NaNs as NaNs for median/MAD
    mask = np.isfinite(data)
    data_nan = np.where(mask, data, np.nan)

    # local median
    if use_numba:
        local_median = nanmedian_filter_numba(data_nan, window=window)
    else:
        local_median = nanmedian_filter(data_nan, window=window)

    # deviations from local medians
    abs_dev = np.abs(data_nan - local_median)

    # Local modified MAD: median of absolute deviations from local median
    if use_numba:
        local_m_mad = nanmedian_filter_numba(abs_dev, window=window)
    else:
        local_m_mad = nanmedian_filter(abs_dev, window=window)

    # convert MAD to sigma-equivalent for normal data 
    # TODO check if this is the right scaling factor for modified MAD
    robust_std = scaling_factor * local_m_mad

    standardized = np.divide(
        data - local_median, 
        robust_std + eps, 
        where=(robust_std + eps > 0), 
        out=np.full_like(data, np.nan)
    )    
    standardized = np.where(np.isfinite(standardized), standardized, np.nan)

    return standardized


@njit
def _nanmad_filter_kernel_numba(padded, H, W, wy, wx):
    """
    Compute MAD (Median Absolute Deviation) at each point.
    For each window centered at (y, x):
      1. Collect finite values in the window
      2. Compute the window's median
      3. Compute median of absolute deviations from that median
    """
    out = np.empty((H, W), dtype=np.float64)
    maxk = wy * wx

    # ---------- PARALLEL ROW PROCESSING ----------
    for y in prange(H):

        # Thread-private working buffers
        values_buf = np.empty(maxk, dtype=np.float64)  # sorted values in window
        abs_dev_buf = np.empty(maxk, dtype=np.float64)  # sorted abs deviations
        size = 0

        # ---- INITIAL WINDOW (x=0) ----
        # Collect finite values from initial window
        size = 0
        for yy in range(y, y + wy):
            for xx in range(0, wx):
                v = padded[yy, xx]
                if not np.isnan(v):
                    idx = _binary_search_numba(values_buf, size, v)
                    # shift right
                    for k in range(size, idx, -1):
                        values_buf[k] = values_buf[k - 1]
                    values_buf[idx] = v
                    size += 1

        # Compute MAD for initial window
        if size > 0:
            # Compute median of this window
            mid = size // 2
            if size % 2 == 1:
                window_median = values_buf[mid]
            else:
                window_median = 0.5 * (values_buf[mid - 1] + values_buf[mid])
            
            # Compute absolute deviations and their median
            abs_dev_size = 0
            for k in range(size):
                dev = np.abs(values_buf[k] - window_median)
                idx = _binary_search_numba(abs_dev_buf, abs_dev_size, dev)
                # shift right
                for m in range(abs_dev_size, idx, -1):
                    abs_dev_buf[m] = abs_dev_buf[m - 1]
                abs_dev_buf[idx] = dev
                abs_dev_size += 1
            
            out[y, 0] = _compute_median_numba(abs_dev_buf, abs_dev_size)
        else:
            out[y, 0] = np.nan

        # ---- SLIDE HORIZONTALLY ----
        for x in range(1, W):

            # --- remove outgoing column (xx=0) ---
            for yy in range(y, y + wy):
                v = padded[yy, x - 1]
                if not np.isnan(v):
                    idx = _binary_search_numba(values_buf, size, v)
                    # remove at idx
                    for k in range(idx, size - 1):
                        values_buf[k] = values_buf[k + 1]
                    size -= 1

            # --- add incoming column (xx=wx-1) ---
            for yy in range(y, y + wy):
                v = padded[yy, x - 1 + wx]
                if not np.isnan(v):
                    idx = _binary_search_numba(values_buf, size, v)
                    # shift to make room
                    for k in range(size, idx, -1):
                        values_buf[k] = values_buf[k - 1]
                    values_buf[idx] = v
                    size += 1

            # Compute MAD for this window
            if size > 0:
                # Compute median of this window
                mid = size // 2
                if size % 2 == 1:
                    window_median = values_buf[mid]
                else:
                    window_median = 0.5 * (values_buf[mid - 1] + values_buf[mid])
                
                # Compute absolute deviations and their median
                abs_dev_size = 0
                for k in range(size):
                    dev = np.abs(values_buf[k] - window_median)
                    idx = _binary_search_numba(abs_dev_buf, abs_dev_size, dev)
                    # shift right
                    for m in range(abs_dev_size, idx, -1):
                        abs_dev_buf[m] = abs_dev_buf[m - 1]
                    abs_dev_buf[idx] = dev
                    abs_dev_size += 1
                
                out[y, x] = _compute_median_numba(abs_dev_buf, abs_dev_size)
            else:
                out[y, x] = np.nan

    return out


def nanmad_filter_numba(data, window):
    """
    NaN-aware MAD (Median Absolute Deviation) filter using Numba acceleration.
    For each point's neighborhood, computes the MAD as:
      1. Median of values in the window
      2. Median of absolute deviations from that median
    """
    # Handle xarray input
    if isinstance(data, xr.DataArray):
        arr = data.values[0]
    else:
        arr = np.asarray(data)

    H, W = arr.shape

    if isinstance(window, int):
        wy = wx = window
    else:
        wy, wx = window

    pad_y = wy // 2
    pad_x = wx // 2

    padded = np.pad(arr,
                    ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant",
                    constant_values=np.nan)

    out = _nanmad_filter_kernel_numba(padded, H, W, wy, wx)

    # Preserve original NaNs
    mask = np.isfinite(arr)
    return np.where(mask, out, np.nan)


def local_standardization_mad_numba(da, window=11, eps=0, scaling_factor=1.4826):
    return local_standardization_mad(da, window=window, eps=eps, scaling_factor=scaling_factor, use_numba=True)


def local_standardization_mad(da, window=11, eps=0, scaling_factor=1.4826, use_numba=True):
    """
    Standard MAD standardization using Numba-accelerated window processing.
     - Local standardization using local median and local MAD (median of absolute deviations from each window's median).
     - Uses nanmad_filter_numba for efficient MAD calculations.
     - Preserves NaNs in the output.
     - Accepts either xarray.DataArray or numpy.ndarray as input.
     - Returns an array with the same shape as input, where each value is standardized based on its local neighborhood defined by the window.
     - eps added to denominator for numerical stability.
     - Scaling factor 1.4826 applied to MAD to convert to sigma-equivalent for normal data.
     - This is the "standard" (not "modified") MAD: MAD is computed independently for each window.
    """
    data = da.values[0] if isinstance(da, xr.DataArray) else np.asarray(da)

    mask = np.isfinite(data)
    data_nan = np.where(mask, data, np.nan)

    # local median
    if use_numba:
        local_median = nanmedian_filter_numba(data_nan, window=window)
    else:
        local_median = nanmedian_filter(data_nan, window=window)

    # local MAD
    if use_numba:
        local_mad = nanmad_filter_numba(data_nan, window=window)
    else:
        def mad_func(window_values):
            m = np.nanmedian(window_values)
            return np.nanmedian(np.abs(window_values - m))

        local_mad = generic_filter(
            data_nan, mad_func, size=window, mode="constant", cval=np.nan
        )

    robust_std = scaling_factor * local_mad

    standardized = (data_nan - local_median) / (robust_std + eps)

    return np.where(np.isfinite(standardized), standardized, np.nan)


def denoise_butterworth(data, cutoff=40, order=2):
    """Applies a Butterworth low-pass filter to the data. This can help to reduce high-frequency noise
    
    """
    from numpy.fft import fft2, ifft2, fftshift, ifftshift

    # TODO make suitable for inputs with very different aspect ratios by allowing different cutoff frequencies for x and y dimensions, and adjusting the kernel accordingly.
    def butterworth_lowpass_kernel(shape, cutoff, order=2):
        ny, nx = shape
        cy, cx = ny // 2, nx // 2
        Y, X = np.ogrid[:ny, :nx]
        D = np.sqrt((Y - cy)**2 + (X - cx)**2)
        
        H = 1 / (1 + (D / cutoff)**(2 * order))
        return H
    
    # accept either DataArray or ndarray
    #data = da.values[0] if isinstance(da, xr.DataArray) else np.asarray(da)

    mask = np.isfinite(data)
    data_filled = np.where(mask, data, 0.0)


    kernel = butterworth_lowpass_kernel(data.shape, cutoff, order)
    data_fft = fft2(data_filled)
    data_fft_shifted = fftshift(data_fft)
    filtered_fft = data_fft_shifted * kernel
    data_filtered = np.real(ifft2(ifftshift(filtered_fft)))

    return np.where(mask, data_filtered, np.nan)
    
    


def combine_standardizations(da, standardizations, coefficients):
    """
    Combines multiple standardized arrays using weighted coefficients.
    """
    # TODO
    raise NotImplementedError("combine_standardizations not implemented yet")



# ==== Wrapper for preprocessing methods ====
def preprocess_data_xr(da, method=None, kwargs=None, variable_names=None, verbose=False):
    """
    
    """
    if kwargs is None:
        kwargs = {}
    
    if method == "local_standardization":
        da[variable_names["value"]].values = local_standardization(da[variable_names["value"]].values, **kwargs)
        return da
    elif method == "local_standardization_mad":
        da[variable_names["value"]].values = local_standardization_mad(da[variable_names["value"]].values, **kwargs)
        return da
    elif method == "local_standardization_m_mad":
        da[variable_names["value"]].values = local_standardization_m_mad(da[variable_names["value"]].values, **kwargs)
        return da
    elif method == "combine_standardizations":
        da[variable_names["value"]].values = combine_standardizations(da[variable_names["value"]].values, **kwargs) # TODO implement
        return da
    elif method == "denoise_butterworth":
        da[variable_names["value"]].values = denoise_butterworth(da[variable_names["value"]].values, **kwargs)
        return da
    elif method == "nan_gaussian_filter":
        da[variable_names["value"]].values = nan_gaussian_filter(da[variable_names["value"]].values, **kwargs)
        return da
    elif method == "crop_to_bbox":
        return crop_to_bbox(da, **kwargs, variable_names=variable_names, verbose=verbose)
    elif method == "quality_filter":
        return quality_filter(da, qa_var=variable_names["quality"], value_var=variable_names["value"], verbose=verbose, **kwargs)
    elif method == "discard_small_files":
        return discard_files_smaller_than(da, variable_name=variable_names["value"], verbose=verbose, **kwargs)
    elif method == "regrid_with_harp": 
        raise NotImplementedError("regrid_with_harp method not implemented yet")
    elif method == "quality_filter":
        raise NotImplementedError("quality_filter method not implemented yet")
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")













# =============================================
#  Cluster analysis utilities
# =============================================

def clusters_plume_id(clusters, file_id, ignore_zero=True) -> np.ndarray:
    """
    Generate plume IDs for each cluster in the clusters array. 
    The plume IDs are generated by concatenating the file_id with a cluster number (starting from 1 if ignore_zero is True, otherwise starting from 0). 
    """
    if ignore_zero: first_index = 1
    else: first_index = 0

    return file_id + "_c" + np.array(range(first_index, int(np.nanmax(clusters))+1), dtype=str) 


def clusters_plume_number(clusters, ignore_zero=True) -> np.ndarray:
    if ignore_zero: first_index = 1
    else: first_index = 0
    return np.array(range(first_index, int(np.nanmax(clusters))+1))


def file_id(filename=None, mode="Sentinel-5P") -> str:

    if mode == "Sentinel-5P":
        # turn 
        # S5P_OFFL_L2__NO2____20240601T002118_20240601T020248_34371_03_020600_20240606T165116.nc
        # into
        # S5P_NO2_240601_34371
        # satellite_product_yymmdd_orbit
        if filename is None:
            return "unknown_file"
        else:
            base = os.path.basename(filename)
            parts = base.split("_")
            if len(parts) < 5:
                raise ValueError(f"Filename {filename} does not match expected format for Sentinel-5p.")
            satellite = parts[0]
            product = parts[4]
            date = parts[8][2:8]  # yymmdd
            orbit = parts[10]
            return f"{satellite}_{product}_{date}_{orbit}"
    else:
        raise ValueError(f"Unknown file_id mode: {mode}")      
    


def clusters_file_name(clusters, file_id, ignore_zero=True) -> np.ndarray:
    n_clusters = int(np.nanmax(clusters))+1
    if ignore_zero:
        n_clusters -= 1
    return np.array([file_id]*n_clusters)


def clusters_n_points_in_file(clusters, return_full_array=True, ignore_zero=True) -> np.ndarray:
    if return_full_array:
        n_points_in_file = np.zeros(int(np.nanmax(clusters))+1, dtype=int) + len(clusters.flatten())
        
        if ignore_zero:
            return n_points_in_file[1:]
        else:
            return n_points_in_file
    
    else:
        return len(clusters.flatten())


def clusters_max_point_index(clusters, values, ignore_zero=True):

    labels = np.asarray(clusters).ravel()
    vals = np.asarray(values).ravel()

    valid = np.isfinite(labels) & np.isfinite(vals)
    labels = labels[valid].astype(np.int64, copy=False)
    vals = vals[valid]
    flat_idx = np.nonzero(valid)[0] # indexes of valid points in flattened array

    if ignore_zero:
        nz = labels > 0
        labels = labels[nz]
        vals = vals[nz]
        flat_idx = flat_idx[nz]

    n_labels = int(np.nanmax(clusters)) + 1

    # Keep same shape idea: one [row, col] per cluster
    out = np.full((n_labels, 2), -1, dtype=np.int64)

    if labels.size == 0:
        return out[1:] if ignore_zero else out

    # Groupwise max via sorting: label asc, value desc
    order = np.lexsort((-vals, labels)) # sort first by label acending, then by value decending. Highest value per label will be first in each group of labels.
    labels_s = labels[order]
    idx_s = flat_idx[order]

    first = np.empty(labels_s.size, dtype=bool)
    first[0] = True
    first[1:] = labels_s[1:] != labels_s[:-1] # find where labels change

    best_labels = labels_s[first]
    best_flat = idx_s[first]

    # Convert flat indices back to (row, col)
    rr, cc = np.unravel_index(best_flat, np.asarray(clusters).shape)
    out[best_labels, 0] = rr
    out[best_labels, 1] = cc

    return out[1:] if ignore_zero else out


def clusters_max_point_locs_from_index(lon, lat, max_point_indices=None, clusters=None, values=None, ignore_zero=None):
    if max_point_indices is None:
        max_point_indices = clusters_max_point_index(clusters, values, ignore_zero)
    return np.transpose((lon[max_point_indices[:, 0], max_point_indices[:, 1]], lat[max_point_indices[:, 0], max_point_indices[:, 1]]))


def clusters_max_point_values_from_index(values, max_point_indices=None, clusters=None, ignore_zero=None):
    if max_point_indices is None:
        max_point_indices = clusters_max_point_index(clusters, values, ignore_zero)
    return values[max_point_indices[:, 0], max_point_indices[:, 1]]


def clusters_timestamp_utc_from_index(timestamps, max_point_indices=None, clusters=None, values=None, ignore_zero=None):
    """ 
    Expects timestamps to be a 1D array of length equal to the number of rows in clusters (= n scanlines).

    """

    if max_point_indices is None:
        max_point_indices = clusters_max_point_index(clusters, values, timestamps, ignore_zero)
    return timestamps[max_point_indices[:, 0]]    



def clusters_n_points(clusters, ignore_zero=True) -> np.ndarray:
    """Count points per cluster label."""
    c = np.asarray(clusters, dtype=float)
    n_labels = int(np.nanmax(c)) + 1
    
    # Keep only valid cluster labels
    valid = np.isfinite(c)
    labels = c[valid].astype(np.int64, copy=False)
    
    # bincount is O(n) and vectorized
    counts = np.bincount(labels, minlength=n_labels)
    
    return counts[1:] if ignore_zero else counts


def clusters_mean_point_values(clusters, values, ignore_zero=True):
    """Mean value per cluster label."""
    c = np.asarray(clusters, dtype=float)
    v = np.asarray(values, dtype=float)
    
    if c.shape != v.shape:
        raise ValueError("clusters and values must have the same shape")
    
    n_labels = int(np.nanmax(c)) + 1
    
    # Keep only valid labels and values
    valid = np.isfinite(c) & np.isfinite(v)
    labels = c[valid].astype(np.int64, copy=False)
    vals = v[valid]
    
    # Sum and counts per label via bincount
    sums = np.bincount(labels, weights=vals, minlength=n_labels)
    counts = np.bincount(labels, minlength=n_labels)
    
    # Safe division sums/counts, avoiding division by zero
    means = np.full(n_labels, np.nan, dtype=float)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    
    return means[1:] if ignore_zero else means

def clusters_median_point_values(clusters, values, ignore_zero=True):
    """
    Median value per cluster label.
    sort labels once, then process grouped slices.
    """
    c = np.asarray(clusters, dtype=float)
    v = np.asarray(values, dtype=float)

    if c.shape != v.shape:
        raise ValueError("clusters and values must have the same shape")

    n_labels = int(np.nanmax(c)) + 1
    out = np.full(n_labels, np.nan, dtype=float)

    # Keep only finite label-value pairs (equivalent to nanmedian behavior per cluster)
    valid = np.isfinite(c) & np.isfinite(v)
    if not np.any(valid):
        return out[1:] if ignore_zero else out

    labels = c[valid].astype(np.int64, copy=False).ravel()
    vals = v[valid].ravel()

    # Group by label via one stable sort
    order = np.argsort(labels, kind="mergesort")
    labels_s = labels[order]
    vals_s = vals[order]

    uniq, starts, counts = np.unique(labels_s, return_index=True, return_counts=True)

    # Median per contiguous group
    for lab, s, cnt in zip(uniq, starts, counts):
        out[lab] = np.median(vals_s[s:s + cnt])

    return out[1:] if ignore_zero else out

def clusters_bounding_boxes(clusters, lon, lat, ignore_zero=True):
    """
    Returns bounding boxes as:
    [min_lon, min_lat, max_lon, max_lat] per cluster label.
    """
    c = np.asarray(clusters, dtype=float)
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)

    if c.shape != x.shape or c.shape != y.shape:
        raise ValueError("clusters, lon, and lat must have the same shape")

    n_labels = int(np.nanmax(c)) + 1

    # Initialize outputs
    min_lon = np.full(n_labels, np.inf, dtype=float) 
    min_lat = np.full(n_labels, np.inf, dtype=float)
    max_lon = np.full(n_labels, -np.inf, dtype=float)
    max_lat = np.full(n_labels, -np.inf, dtype=float)

    # Keep only finite triplets
    valid = np.isfinite(c) & np.isfinite(x) & np.isfinite(y)
    labels = c[valid].astype(np.int64, copy=False)
    xv = x[valid]
    yv = y[valid]

    # Optional: skip label 0 directly
    if ignore_zero:
        keep = labels > 0
        labels = labels[keep]
        xv = xv[keep]
        yv = yv[keep]

    # One-pass grouped min/max
    np.minimum.at(min_lon, labels, xv) # index=labels, values=xv
    np.minimum.at(min_lat, labels, yv) 
    np.maximum.at(max_lon, labels, xv)
    np.maximum.at(max_lat, labels, yv)

    bboxes = np.column_stack((min_lon, min_lat, max_lon, max_lat))

    # Labels not present remain inf/-inf, convert to NaN
    missing = ~np.isfinite(min_lon) | ~np.isfinite(min_lat) | ~np.isfinite(max_lon) | ~np.isfinite(max_lat)
    bboxes[missing] = np.nan

    return bboxes[1:] if ignore_zero else bboxes


def calc_pixel_areas_km2(lon_bounds, lat_bounds, geod=Geod(ellps="WGS84")):
    """
    lon_bounds, lat_bounds: shape (n_pixels, 4)
    returns: area_km2, shape (n_pixels,)
    """
    shape = lon_bounds.shape[:-1]
    lon_bounds = lon_bounds.reshape(-1, 4)
    lat_bounds = lat_bounds.reshape(-1, 4)
    n = lon_bounds.shape[0]
    areas = np.empty(n, dtype=np.float64)

    for i in range(n):
        area, _ = geod.polygon_area_perimeter(
            lon_bounds[i], lat_bounds[i]
        )
        areas[i] = abs(area) * 1e-6  # m² → km²

    return areas.reshape(shape)
def cluster_areas_from_pixel_areas(
    clusters, lon_bounds, lat_bounds, ignore_zero=True
):
    """
    clusters: shape (n_pixels,), integer labels
    pixel_areas_km2: shape (n_pixels,)
    """

    # Flatten
    clusters = clusters.reshape(-1)
    lon_bounds = lon_bounds.reshape(-1, 4)
    lat_bounds = lat_bounds.reshape(-1, 4)

    #

    geod = Geod(ellps="WGS84")

    if ignore_zero:
        clusters = clusters.astype(np.int64)
        valid = np.isfinite(clusters) & (clusters > 0)
        lon_bounds = lon_bounds[valid]
        lat_bounds = lat_bounds[valid]
        clusters = clusters[valid]
        mask = np.isfinite(clusters)
    else:
        mask = np.isfinite(clusters)
        lon_bounds = lon_bounds[mask]
        lat_bounds = lat_bounds[mask]
        clusters = clusters[mask].astype(np.int64)

    pixel_areas_km2 = calc_pixel_areas_km2(lon_bounds, lat_bounds, geod=geod)

    areas = np.bincount(
        clusters.astype(np.int64).flatten(),
        weights=pixel_areas_km2.flatten()
    )

    if ignore_zero:
        areas = areas[1:]

    return areas

def cluster_areas_fast_approximate_sentinel5():
    # TODO implement a fast approximation of cluster areas for Sentinel-5P data, based on the known pixel size at the equator and the latitude of the cluster (since pixel size varies with latitude).
    raise NotImplementedError("cluster_areas_fast_approximate_sentinel5 not implemented yet")


def clusters_is_weekend(time, clusters=None, return_full_array=True, ignore_zero=True):
    """Whether each timestamp falls on a weekend (Sat/Sun)."""
    time_array = pd.to_datetime(time).tz_localize(None).values
    ts_days = np.asarray(time_array, dtype="datetime64[D]")

    result = ~np.is_busday(ts_days)[0]  # True for Sat/Sun

    if return_full_array:
        if ignore_zero:
            return np.zeros(int(np.nanmax(clusters)),dtype=bool) + result
        else:
            return np.zeros(int(np.nanmax(clusters))+1,dtype=bool) + result
    else:
        return result[0]
    
def clusters_day_of_week(time, clusters=None, return_full_array=True, ignore_zero=True):
    """
    Day of week from timestamps, Monday=0 ... Sunday=6.

    If return_full_array=True this returns an array of length
    `int(np.nanmax(clusters)) + 1`. If `ignore_zero=True` the returned
    array slices off index 0 so its length is `n_labels - 1`.
    """
    time_array = pd.to_datetime(time).tz_localize(None).values
    ts_days = np.asarray(time_array, dtype="datetime64[D]")

    # Monday=0 ... Sunday=6
    dow = (ts_days.astype("int64") + 3)[0] % 7

    if return_full_array:
        if clusters is None:
            raise ValueError("clusters must be provided when return_full_array=True")
        n = int(np.nanmax(clusters)) + 1

        if dow.ndim == 0:
            arr = np.full(n, int(dow), dtype=np.int8)
        elif dow.size == 1:
            arr = np.full(n, int(dow.ravel()[0]), dtype=np.int8)
        else:
            arr = dow.astype(np.int8, copy=False)

        return arr[1:] if ignore_zero else arr

    if dow.ndim == 0:
        return int(dow)
    return dow[0].astype(np.int8, copy=False)


def clusters_is_land(max_point_locs, map_data=None):
    """Vectorized spatial join for checking if points are on land."""
    if map_data is None:
        map_data = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        )
        map_data = map_data.to_crs(epsg=4326)

    # Create GeoDataFrame from points
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in max_point_locs],
        crs="EPSG:4326"
    )
    
    # Spatial join: points that intersect with map polygons are on land
    result = gpd.sjoin(points_gdf, map_data, how='left', predicate='intersects')
    is_land_array = result['index_right'].notna().values
    
    return is_land_array




def build_latlon_kdtree(lon_grid, lat_grid, values):
    lons = lon_grid.ravel()
    lats = lat_grid.ravel()
    vals = values.ravel()

    valid = ~np.isnan(vals)
    lons = lons[valid]
    lats = lats[valid]
    vals = vals[valid]

    # Scale longitude by cos(latitude)
    x = lons * np.cos(np.deg2rad(lats))
    y = lats

    tree = KDTree(np.column_stack((x, y)))

    return tree, lons, lats, vals


def clusters_median_of_neighbourhood_kdtree_exact(
    tree,
    lons,
    lats,
    vals,
    max_point_locs,
    radius_km
):
    medians = np.full(len(max_point_locs), np.nan)

    radius_deg = radius_km / 111.32  # rough upper bound

    for i, (lon_c, lat_c) in enumerate(max_point_locs):
        xc = lon_c * np.cos(np.deg2rad(lat_c))
        yc = lat_c

        idx = tree.query_ball_point([xc, yc], radius_deg)
        if not idx:
            continue

        candidates = np.column_stack((lats[idx], lons[idx]))
        center = np.column_stack((
            np.full(len(idx), lat_c),
            np.full(len(idx), lon_c),
        ))

        dists = haversine_vector(candidates, center, Unit.KILOMETERS)

        neighbors = vals[idx][dists <= radius_km]
        if neighbors.size > 0:
            medians[i] = np.nanmedian(neighbors)

    return medians


def clusters_is_max_point_on_edge(clusters, max_point_indices, ignore_zero=True):
    is_on_edge = np.zeros(len(max_point_indices), dtype=bool)

    for i, (row_idx, col_idx) in enumerate(max_point_indices):
        cluster_id = i + 1 if ignore_zero else i
        neighbors = clusters[max(0, row_idx-1):row_idx+2, max(0, col_idx-1):col_idx+2].flatten()
        is_on_edge[i] = np.any((neighbors != cluster_id) & ~np.isnan(neighbors))

    return is_on_edge



def clusters_is_plume_connected_to_nans_or_edge(
    clusters,
    ignore_zero=True,
    connectivity=8,   # 4 or 8
):
    c = np.asarray(clusters, dtype=float)
    n_clusters = int(np.nanmax(c)) + 1
    out = np.zeros(n_clusters, dtype=bool)

    # Pad with NaN so touching the image boundary counts as touching edge
    p = np.pad(c, 1, mode="constant", constant_values=np.nan)
    center = p[1:-1, 1:-1]

    if connectivity == 4:
        # N, S, W, E
        touch_nan = (
            np.isnan(p[:-2, 1:-1]) |   # north
            np.isnan(p[2:, 1:-1])  |   # south
            np.isnan(p[1:-1, :-2]) |   # west
            np.isnan(p[1:-1, 2:])       # east
        )
    elif connectivity == 8:
        # N, S, W, E + diagonals
        touch_nan = (
            np.isnan(p[:-2, :-2]) | np.isnan(p[:-2, 1:-1]) | np.isnan(p[:-2, 2:]) |
            np.isnan(p[1:-1, :-2])                           | np.isnan(p[1:-1, 2:]) |
            np.isnan(p[2:, :-2])  | np.isnan(p[2:, 1:-1])  | np.isnan(p[2:, 2:])
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    valid = np.isfinite(center)
    if ignore_zero:
        valid &= (center > 0)

    touched_labels = center[valid & touch_nan].astype(np.int64)
    if touched_labels.size:
        # Slightly faster than unique for this use case
        hit = np.bincount(touched_labels, minlength=n_clusters) > 0
        out |= hit

    return out[1:] if ignore_zero else out



def clusters_connected_plumes(clusters, ignore_zero=True, connectivity=8):
    """
    For each cluster, return a set of adjacent cluster IDs.
    connectivity: 4 or 8

    Behavior with ignore_zero:
    - ignore_zero=False: returns sets for clusters 0..N
    - ignore_zero=True: returns sets for clusters 1..N
      but 0 is still included inside those sets if adjacent
    """
    c = np.asarray(clusters, dtype=float)
    n_all = int(np.nanmax(c)) + 1

    # Output layout
    n_out = n_all - 1 if ignore_zero else n_all
    connected = [set() for _ in range(n_out)]

    def collect_pairs(a, b):
        # Finite neighboring labels that differ
        valid = np.isfinite(a) & np.isfinite(b) & (a != b)

        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64)

        p = np.stack(
            [a[valid].astype(np.int64), b[valid].astype(np.int64)],
            axis=1
        )

        # Undirected edge: (i, j) == (j, i)
        p.sort(axis=1)

        # Remove duplicates
        return np.unique(p, axis=0)

    # 4-neighborhood: horizontal + vertical
    pair_blocks = [
        collect_pairs(c[:, :-1], c[:, 1:]),   # left-right
        collect_pairs(c[:-1, :], c[1:, :])    # up-down
    ]

    # 8-neighborhood adds diagonals
    if connectivity == 8:
        pair_blocks.append(collect_pairs(c[:-1, :-1], c[1:, 1:]))  # down-right
        pair_blocks.append(collect_pairs(c[:-1, 1:], c[1:, :-1]))  # down-left
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8")

    pairs = np.concatenate(pair_blocks, axis=0)
    if pairs.size == 0:
        return connected

    pairs = np.unique(pairs, axis=0)

    def idx_in_output(label):
        if ignore_zero:
            if label == 0:
                return None
            return label - 1
        return label

    # Build symmetric adjacency sets
    for a, b in pairs:
        ia = idx_in_output(a)
        ib = idx_in_output(b)

        # Add b into a's set if a has an output slot
        if ia is not None:
            connected[ia].add(int(b))

        # Add a into b's set if b has an output slot
        if ib is not None:
            connected[ib].add(int(a))

    return connected


def clusters_mean_q_value(clusters, qa_values, ignore_zero=True):
    """
    Mean qa_value per cluster label.

    Parameters
    ----------
    clusters : array-like
        Cluster label grid (can contain NaN). Expected integer-like labels: 0,1,2,...
    qa_values : array-like
        qa_value grid, same shape as clusters.
    ignore_zero : bool
        If True, exclude cluster 0 from returned array.

    Returns
    -------
    np.ndarray
        Mean qa_value per cluster.
        - If ignore_zero=False: index i corresponds to cluster i.
        - If ignore_zero=True: result[0] corresponds to cluster 1.
        Clusters with no valid qa pixels are returned as NaN.
    """
    c = np.asarray(clusters, dtype=float)
    q = np.asarray(qa_values, dtype=float)

    if c.shape != q.shape:
        raise ValueError("clusters and qa_values must have the same shape")

    # Keep only finite labels and finite qa values
    valid = np.isfinite(c) & np.isfinite(q)
    if not np.any(valid):
        return np.array([], dtype=float)

    labels = c[valid].astype(np.int64, copy=False)
    qa = q[valid]

    n_labels = int(np.nanmax(c)) + 1

    # Sum and counts per label
    sums = np.bincount(labels, weights=qa, minlength=n_labels)
    counts = np.bincount(labels, minlength=n_labels)

    means = np.full(n_labels, np.nan, dtype=float)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    return means[1:] if ignore_zero else means


# Wrapper for cluster analysis methods
def analyze_clusters(method, clusters, lon=None, lat=None, values=None, timestamps=None, qa_values=None, variable_names=None, max_point_indices=None, ignore_zero=True, verbose=False, filename=None, **kwargs):
    """ """

    if method == "plume_id":
        return clusters_plume_id(clusters, file_id=file_id(filename=filename), ignore_zero=ignore_zero)
    elif method == "plume_number":
        return clusters_plume_number(clusters, ignore_zero=ignore_zero)
    elif method == "file_name":
        return clusters_file_name(clusters, file_id=file_id(filename=filename), ignore_zero=ignore_zero)
    elif method == "n_points_in_file":
        return clusters_n_points_in_file(clusters, return_full_array=False, ignore_zero=ignore_zero)
    elif method == "max_point_indices":
        return clusters_max_point_index(clusters, values, ignore_zero=ignore_zero)
    elif method == "max_point_locs":
        return clusters_max_point_locs_from_index(lon, lat, max_point_indices=max_point_indices, clusters=clusters, values=values, ignore_zero=ignore_zero)
    elif method == "max_point_value":
        return clusters_max_point_values_from_index(values, max_point_indices=max_point_indices, clusters=clusters, ignore_zero=ignore_zero)
    elif method == "timestamp_utc":
        return clusters_timestamp_utc_from_index(timestamps, max_point_indices=max_point_indices, clusters=clusters, values=values, ignore_zero=ignore_zero)
    elif method == "n_points":
        return clusters_n_points(clusters, ignore_zero=ignore_zero)
    elif method == "mean_point_value":
        return clusters_mean_point_values(clusters, values, ignore_zero=ignore_zero)
    elif method == "median_point_value":
        return clusters_median_point_values(clusters, values, ignore_zero=ignore_zero)
    elif method == "bounding_box":
        return clusters_bounding_boxes(clusters, lon, lat, ignore_zero=ignore_zero)
    elif method == "areas_from_pixel_areas":
        raise NotImplementedError("areas_from_pixel_areas not implemented yet")
    elif method == "is_weekend":
        return clusters_is_weekend(timestamps, clusters=clusters, return_full_array=True, ignore_zero=ignore_zero)
    elif method == "day_of_week":
        return clusters_day_of_week(timestamps, clusters=clusters, return_full_array=True, ignore_zero=ignore_zero)
    elif method == "is_on_land":
        max_point_locs = clusters_max_point_locs_from_index(lon, lat, max_point_indices=max_point_indices, clusters=clusters, values=values, ignore_zero=ignore_zero)
        return clusters_is_land(max_point_locs)
    elif method == "median_of_neighbourhood":
        tree, lons, lats, vals = build_latlon_kdtree(lon, lat, values)
        max_point_locs = clusters_max_point_locs_from_index(lon, lat, max_point_indices=max_point_indices, clusters=clusters, values=values, ignore_zero=ignore_zero)
        radius_km = kwargs.get("radius_km", 10)
        return clusters_median_of_neighbourhood_kdtree_exact(tree, lons, lats, vals, max_point_locs, radius_km)
    elif method == "is_max_point_on_edge":
        return clusters_is_max_point_on_edge(clusters, max_point_indices, ignore_zero)
    elif method == "is_plume_connected_to_nans_or_edge":
        return clusters_is_plume_connected_to_nans_or_edge(clusters, ignore_zero, connectivity=kwargs.get("connectivity", 8))
    elif method == "connected_plumes":
        return clusters_connected_plumes(clusters, ignore_zero, connectivity=kwargs.get("connectivity", 8))
    elif method == "mean_q_value":
        return clusters_mean_q_value(clusters, qa_values, ignore_zero)
    else:
        raise ValueError(f"Unknown cluster analysis method: {method}")


    

# ==================================================================================================================================================================================================================================================


def scea_interactive(
        lon, lat, value,  
        windows_list = [9, 11, 13, 15, 21, 31, 51, 101, 151, 201, 301, 501],
        wind_data=None,
        use_numba=True,
):
    
    # Interactive multi-scale sliders + optional SCEA run (extended with per-scale denoise)
    from ipywidgets import IntSlider, FloatSlider, Button, HBox, Layout, VBox, Output, Dropdown, Checkbox, FloatText, IntText, Label, AppLayout, BoundedFloatText
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print(""" TODO
Instructions:

    PRE-PROSESSING:
          
    SCEA PARAMETERS:
        growth_limit: Higher value -> smaller clusters. The number sets a threshold in terms of standard deviations from the median (or mean) of the local window. Points with values below the threshold will not enlargen the cluster.
        detection_limit: Higher value -> less clusters. The number sets a threshold in terms of standard deviations from the median (or mean) for the initial seed points.
        local_box_radius: Radius of the local window (in coordinate units) that are considered for each seed point when creating a cluster. Larger windows are computationally more expensive.
        max_pts_start_radius: Approximately, larger value -> jumps bigger gaps. The number sets a threshold for the maximum numbe of clustered points that can be within the starting radius of a point. If there are more points than this threshold, the point will not enlargen the cluster.
          """)

    compact_layout = Layout(width='130px')
    less_compact_layout = Layout(width='140px')
    full_desc = {"description_width": "initial"}
    scea_parameters_layout_float = Layout(width='160px')  # top and bottom margin for spacing
    scea_parameters_layout_int = Layout(width='180px')  # top and bottom margin for spacing

    out = Output()
    #last = {"combined": None, "clusters": None}
    standardized_cache = {}

    last = {"combined": None}
    cluster_store = {}                  # {"clusters_1": array, "clusters_2": array, ...}
    cluster_counter = {"n": 0}          # mutable counter for closures

    cluster_select = Dropdown(
        options=[("None", "none")],
        value="none",
        description="Clusters:",
        layout=Layout(width='240px')
    )

    std_type_options = [("standardization", "standard"), ("median_MAD", "median_mad")]
    std_type_small  = Dropdown(options=std_type_options, value="standard", description="std_type", layout=less_compact_layout)
    std_type_medium = Dropdown(options=std_type_options, value="standard", description="std_type", layout=less_compact_layout)
    std_type_large  = Dropdown(options=std_type_options, value="standard", description="std_type", layout=less_compact_layout)

    # --- scale/window controls ---
    w_small = Dropdown(options=windows_list, value=9, description="window_s", layout=less_compact_layout)
    w_med   = Dropdown(options=windows_list, value=13, description="window_s", layout=less_compact_layout)
    w_large = Dropdown(options=windows_list, value=501, description="window_s", layout=less_compact_layout)

    c_small = FloatText(value=1.0, step=0.1, description="coef", layout=compact_layout)
    c_med   = FloatText(value=1.0, step=0.1, description="coef", layout=compact_layout)
    c_large = FloatText(value=1.0, step=0.1, description="coef", layout=compact_layout)


    use_small = Checkbox(value=True, description="SMALL WINDOW", layout=Layout(width='220px'))
    use_medium = Checkbox(value=True, description="MEDIUM WINDOW", layout=Layout(width='220px'))
    use_large = Checkbox(value=True, description="LARGE WINDOW", layout=Layout(width='220px'))
    use_raw = Checkbox(value=False, description="RAW DATA", layout=Layout(width='220px'))

    # --- per-scale pre-denoise controls ---
    denoise_pre_small = Checkbox(value=False, description="denoise_pre", layout=Layout(width='210px'))
    denoise_pre_medium = Checkbox(value=False, description="denoise_pre", layout=Layout(width='210px'))
    denoise_pre_large = Checkbox(value=True, description="denoise_pre", layout=Layout(width='210px'))

    sigma_pre_small = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)
    sigma_pre_medium = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)
    sigma_pre_large = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)

    # --- per-scale post-denoise controls ---
    denoise_post_small = Checkbox(value=False, description="denoise_post", layout=Layout(width='210px'))
    denoise_post_medium = Checkbox(value=True, description="denoise_post", layout=Layout(width='210px'))
    denoise_post_large = Checkbox(value=False, description="denoise_post", layout=Layout(width='210px'))

    sigma_post_small = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)
    sigma_post_medium = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)
    sigma_post_large = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)

    # --- overlay controls ---
    show_clusters = Checkbox(value=False, description="overlay clusters", layout=Layout(width='210px'))
    overlay_option = Dropdown(
        options=[
            ('None', 'none'),
            ('Raw', 'raw'),
            ('Combined Standardized', 'combined'),
            ('Small Only', 'small'),
            ('Medium Only', 'medium'),
            ('Large Only', 'large'),
        ],
        value='combined',
        description='OVERLAY: ',
        layout=Layout(width='240px')
    )
    vmax_q = BoundedFloatText(value=0.999, min=0.0, max=1.0, step=0.001, description="vmax_q", layout=Layout(width="145px"))


    # --- raw data controls ---
    denoise_raw = Checkbox(value=False, description="denoise_raw", layout=Layout(width='210px'))
    sigma_raw = FloatText(value=1.0, step=0.2, description="sigma", layout=compact_layout)
    c_raw = FloatText(value=1.0, step=0.1, description="coef", layout=compact_layout)

    # --- SCEA controls (text boxes) ---
    scea_label = Label(
        "SCEA PARAMETERS:",
        layout=Layout(margin="0 10px 0 0", width="150px")  # right margin adds space after text
    )
    growth_limit_w = FloatText(value=2.1, step=0.2, description="growth_limit", tooltip="asdf asdf asdf", layout=scea_parameters_layout_float, style=full_desc)
    detection_limit_w = FloatText(value=2.5, step=0.2, description="detection_limit", layout=scea_parameters_layout_float, style=full_desc)
    local_box_size_w = IntText(value=4, description="local_box_radius", layout=scea_parameters_layout_int, style=full_desc)
    max_pts_start_radius_w = IntText(value=5, description="max_pts_start_radius", layout=scea_parameters_layout_int, style=full_desc)
    run_btn = Button(description="Run SCEA", button_style="warning", style=full_desc, tooltip=" asdf asdf asdf s")

    # --- wind controls ---
    use_wind = Checkbox(value=True, description="Use wind in SCEA")
    wind_level = Dropdown(
        options=[("100m", "100m"), ("10m", "10m")],
        value="100m",
        description="wind level",
        layout=scea_parameters_layout_float,
    )
    wind_scale_method = Dropdown(
        options=[("linear", "linear"), ("sqrt", "sqrt"), ("log", "log")],
        value="linear",
        description="wind scale",
        layout=scea_parameters_layout_float,
    )
    wind_scale_param = FloatText(value=0.05, step=0.01, description="wind param", layout=scea_parameters_layout_float)

    overlay_wind_arrows = Checkbox(value=False, description="overlay wind arrows", layout=Layout(width='230px'))

    def apply_denoise(arr, sigma=1.5):
        return nan_gaussian_filter(arr, sigma=float(sigma))

    def get_local_standardization(window, std_type_val, denoise_pre_flag, sigma_pre_val, denoise_post_flag, sigma_post_val):
        key = (
            int(window),
            str(std_type_val),
            bool(denoise_pre_flag),
            round(float(sigma_pre_val), 3),
            bool(denoise_post_flag),
            round(float(sigma_post_val), 3),
        )
        if key in standardized_cache:
            return standardized_cache[key]

        base = value
        
        # Pre-denoise
        if denoise_pre_flag:
            base = apply_denoise(base, sigma=sigma_pre_val)
        
        # Standardize
        if std_type_val == "median_mad":
            arr = np.asarray(local_standardization_m_mad(base, window=int(window), use_numba=use_numba))
        else:
            arr = np.asarray(local_standardization(base, window=int(window)))

        # Post-denoise
        if denoise_post_flag:
            arr = apply_denoise(arr, sigma=sigma_post_val)
        
        standardized_cache[key] = arr
        return arr


    def render_combined():
        s = get_local_standardization(
            w_small.value, std_type_small.value,
            denoise_pre_small.value, sigma_pre_small.value,
            denoise_post_small.value, sigma_post_small.value
        )
        m = get_local_standardization(
            w_med.value, std_type_medium.value,
            denoise_pre_medium.value, sigma_pre_medium.value,
            denoise_post_medium.value, sigma_post_medium.value
        )
        l = get_local_standardization(
            w_large.value, std_type_large.value,
            denoise_pre_large.value, sigma_pre_large.value,
            denoise_post_large.value, sigma_post_large.value
        )
        
        # Raw data (optionally denoised)
        raw = value
        if denoise_raw.value:
            raw = apply_denoise(raw, sigma=sigma_raw.value)
        # Standardize raw to match the scale of s/m/l
        raw = (raw - np.nanmean(raw)) / (np.nanstd(raw) + 1e-12)

        combined = (
            (float(c_small.value) if use_small.value else 0.0) * s
            + (float(c_med.value) if use_medium.value else 0.0) * m
            + (float(c_large.value) if use_large.value else 0.0) * l
            + (float(c_raw.value) if use_raw.value else 0.0) * raw
        )

        last["combined"] = combined
        return combined

    def update_plot(*_):
        
        combined = render_combined()

        # Get standardized data for each scale
        s = get_local_standardization(
            w_small.value, std_type_small.value,
            denoise_pre_small.value, sigma_pre_small.value,
            denoise_post_small.value, sigma_post_small.value
        )
        m = get_local_standardization(
            w_med.value, std_type_medium.value,
            denoise_pre_medium.value, sigma_pre_medium.value,
            denoise_post_medium.value, sigma_post_medium.value
        )
        l = get_local_standardization(
            w_large.value, std_type_large.value,
            denoise_pre_large.value, sigma_pre_large.value,
            denoise_post_large.value, sigma_post_large.value
        )
        
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(1, 1, figsize=(14, 8), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=150)
            
            # Determine what to display based on overlay_option
            if overlay_option.value == 'raw':
                display_data = value
                title_suffix = "Raw Data"
            elif overlay_option.value == 'small':
                display_data = s
                title_suffix = f"Small (w={w_small.value})"
            elif overlay_option.value == 'medium':
                display_data = m
                title_suffix = f"Medium (w={w_med.value})"
            elif overlay_option.value == 'large':
                display_data = l
                title_suffix = f"Large (w={w_large.value})"
            else:  # 'combined' or 'none'
                display_data = combined
                title_suffix = "Standardization windows combined"
            
            # Only plot if not 'none'
            if overlay_option.value != 'none':
                pcm = ax.pcolormesh(
                    lon, lat, display_data,
                    shading="auto",
                    vmax=np.nanquantile(display_data[~np.isnan(display_data)], q=float(vmax_q.value)),
                    transform=ccrs.PlateCarree(),
                    cmap="cmc.batlow"
                )
                plt.colorbar(pcm, ax=ax)
            
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.5)
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.5)
            
            # Overlay clusters if enabled and available
            selected_key = cluster_select.value
            selected_clusters = cluster_store.get(selected_key) if selected_key != "none" else None

            if show_clusters.value and selected_clusters is not None:
                ax.pcolormesh(
                    lon, lat,
                    np.where(selected_clusters > 0, selected_clusters, np.nan),
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                    cmap=cc.cm.glasbey_light,
                    alpha=1,
                )
                ax.set_title(f"{title_suffix} + {selected_key}")
            else:
                ax.set_title(f"{title_suffix}")
                    
            if overlay_wind_arrows.value:
                angle_key = f"wind_angle_{wind_level.value}"
                mag_key = f"wind_magnitude_{wind_level.value}"

                #wind_data_tropomi = match_era5_wind_to_tropomi(tropomi_data_cropped, era5_data)

                angle = wind_data[angle_key].data
                mag = wind_data[mag_key].data

                # Apply same preprocessing used for SCEA
                mag_scaled = scale_wind_magnitude_for_distance_matrix(
                    mag.flatten(),
                    method=wind_scale_method.value,
                    parameter=float(wind_scale_param.value),
                ).reshape(mag.shape)

                skip = (slice(None, None, 3), slice(None, None, 3))
                u = mag_scaled * np.cos(angle)
                v = mag_scaled * np.sin(angle)

                ax.quiver(
                    lon[skip], lat[skip],
                    u[skip], v[skip],
                    transform=ccrs.PlateCarree(),
                    color="black",
                    alpha=0.45,
                    scale=10,
                    width=0.001,
                    headaxislength=3,
                )

            plt.show()

    def mark_clusters_stale(*_):
        run_btn.button_style = "warning"
        run_btn.description = "Run SCEA (stale)"
        update_plot()

    def run_scea_clicked(b):
        show_clusters.value = False
        b.description = "Running..."
        b.disabled = True
        b.button_style = "info"

        combined = last.get("combined") if last.get("combined") is not None else render_combined()

        point_coordinates = [lon, lat]

        # default SCEA args (no wind)
        scea_kwargs = dict(
            coords=point_coordinates,
            values=combined,
            growth_limit=float(growth_limit_w.value),
            detection_limit=float(detection_limit_w.value),
            local_box_size=int(local_box_size_w.value),
            max_pts_start_radius=int(max_pts_start_radius_w.value),
            metric="geodesic",
        )

        if use_wind.value:
            # expects 'era5_data' and match_era5_wind_to_tropomi(...) available from earlier cells

            angle_key = f"wind_angle_{wind_level.value}"
            mag_key = f"wind_magnitude_{wind_level.value}"

            scaled_mag = scale_wind_magnitude_for_distance_matrix(
                wind_data[mag_key].data.flatten(),
                method=wind_scale_method.value,
                parameter=float(wind_scale_param.value),
            )

            scea_kwargs["metric"] = "geodesic_half_ellipse"
            scea_kwargs["distance_matrix_kwargs"] = {
                "rotation": wind_data[angle_key].data.flatten(),
                "magnitude": scaled_mag,
            }

        clusters = scea(**scea_kwargs)

        cluster_counter["n"] += 1
        cluster_name = f"clusters_{cluster_counter['n']}"
        cluster_store[cluster_name] = np.array(clusters, copy=True)

        # Refresh dropdown options and select newest result
        cluster_select.options = [("None", "none")] + [(k, k) for k in cluster_store.keys()]
        cluster_select.value = cluster_name

        #show_clusters.value = True

        b.description = "Run SCEA"
        b.button_style = "success"
        b.disabled = False
        show_clusters.value = True

    # Register all widget observers BEFORE initial display
    for widget in (
        w_small, w_med, w_large, c_small, c_med, c_large,
        std_type_small, std_type_medium, std_type_large,
        use_small, use_medium, use_large, use_raw,
        denoise_pre_small, denoise_pre_medium, denoise_pre_large,
        sigma_pre_small, sigma_pre_medium, sigma_pre_large,
        denoise_post_small, denoise_post_medium, denoise_post_large,
        sigma_post_small, sigma_post_medium, sigma_post_large,
        denoise_raw, sigma_raw, c_raw,
        use_wind, wind_level, wind_scale_method, wind_scale_param,
    ):
        widget.observe(mark_clusters_stale, names="value")

    # Overlay and cluster visibility changes
    overlay_option.observe(update_plot, names="value")
    vmax_q.observe(update_plot, names="value")
    show_clusters.observe(update_plot, names="value")
    overlay_wind_arrows.observe(update_plot, names="value")
    cluster_select.observe(update_plot, names="value")

    # SCEA params invalidate clusters
    for widget in (growth_limit_w, detection_limit_w, local_box_size_w, max_pts_start_radius_w):
        widget.observe(mark_clusters_stale, names="value")

    # Register button click
    run_btn.on_click(run_scea_clicked)

    center_layout = Layout(align_items='center', justify_content='center', background_color='lightblue')
    parameters_layout = Layout(display='flex', flex_flow='', align_items='flex-start', justify_content='flex-start',  padding='5px')
    parameters_layout = Layout(
        display="flex",
        flex_flow="row wrap",
        align_items="center",
        justify_content="flex-start",
        gap="8px",
        padding="0px",
    )
    row_layout = Layout(
        justify_content="flex-end",
        align_items="center",
        gap="0px",                     # small explicit gap
        margin="0",
        padding="0",
        width="fit-content",
    )
    section_layout = Layout(
        border="1px solid gray",
        padding="0px 0px",
        margin="0",
        width="285px",      # shrink section to children
        align_items="flex-end",
    )
    scales_row_layout = Layout(
        justify_content="flex-start",
        align_items="flex-start",
        gap="8px",
        width="fit-content",
    )
    controls_layout = Layout(
        align_items="flex-start",
        gap="6px",
    )

    # Controls layout
    controls = VBox([
        Label("DATA PRE-PROCESSING: Adaptive standardization with multiple scales and optional denoising", layout=Layout(padding="0px 0px 0px 5px")),
        HBox([
            VBox([
                HBox([use_small, std_type_small], layout=row_layout),
                HBox([denoise_pre_small, sigma_pre_small], layout=row_layout),
                HBox([w_small, c_small], layout=row_layout),
                HBox([denoise_post_small, sigma_post_small], layout=row_layout),
            ], layout=section_layout),
            VBox([
                HBox([use_medium, std_type_medium], layout=row_layout),
                HBox([denoise_pre_medium, sigma_pre_medium], layout=row_layout),
                HBox([w_med, c_med], layout=row_layout),
                HBox([denoise_post_medium, sigma_post_medium], layout=row_layout),
            ], layout=section_layout),
            VBox([
                HBox([use_large, std_type_large], layout=row_layout),
                HBox([denoise_pre_large, sigma_pre_large], layout=row_layout),
                HBox([w_large, c_large], layout=row_layout),
                HBox([denoise_post_large, sigma_post_large], layout=row_layout),
            ], layout=section_layout),
            VBox([
                HBox([use_raw], layout=row_layout),
                HBox([denoise_raw], layout=row_layout),
                HBox([sigma_raw], layout=row_layout),  # spacer to align with window rows
                HBox([c_raw], layout=row_layout),
            ], layout=Layout(border="1px solid gray",padding="0px 0px",margin="0",width="155px",align_items="flex-end",)),
        ], layout=scales_row_layout),

        HBox([scea_label, growth_limit_w, detection_limit_w, local_box_size_w, max_pts_start_radius_w, run_btn], layout=parameters_layout),
        HBox([use_wind, wind_level, wind_scale_method, wind_scale_param], layout=parameters_layout),
        HBox([overlay_option, vmax_q, show_clusters, cluster_select, overlay_wind_arrows], layout=Layout(padding="10px", display='flex', flex_flow='row wrap', align_items='center', justify_content='flex-start', gap='0px')),
    ], layout=controls_layout)

    # Display UI and initial plot
    display(controls, out)
    update_plot()