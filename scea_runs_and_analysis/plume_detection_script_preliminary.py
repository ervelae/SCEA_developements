import os
import datetime
from glob import glob
import hashlib
import json
import yaml
import numpy as np
import pandas as pd
import sys
import copy
import xarray as xr
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import time
import gc
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.collections import PatchCollection
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    try:
        import cmcrameri.cm as cmc
    except Exception:
        cmc = None
    try:
        import colorcet as cc
    except Exception:
        cc = None
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    Normalize = None
    Patch = None
    ccrs = None
    cfeature = None
    cmc = None
    cc = None
    MATPLOTLIB_AVAILABLE = False
else:
    plt.ion()



def default_config():
    return {        
        # File variables
        "io_variables": {
            #"files_dir": '/Users/eliaserv/Documents/Satellite data/sentinel5p-no2/2024/S5P_NO2_January_2024',
            "files_dir": '/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA_developements/example_data',
            "files_type": "nc",
            "file_indices_to_extract": [1,2,3], # e.g. [0, 1, 2] to extract first three files, or None to extract all files
            "open_file_kwargs": {"group": "PRODUCT", "mask_and_scale": True}, # e.g. {"chunks": {"time": 10}} to open files with dask and chunking, or {} for default xarray open_dataset behavior
            "wind_data_dir": '/Volumes/One Touch/ERA5/data.grib',
            "wind_data_type": "grib",
            "output_dir": "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA_developements/scea_runs_and_analysis/outputs",
            "scea_path": "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA_developements/SCEA_v3.py"
        },

        # Variable names in the files
        "variable_names": {
            "value": "nitrogendioxide_tropospheric_column", 
            "lat": "latitude", 
            "lon": "longitude", 
            "time": "time_utc", 
            "quality": "qa_value"
        }, 

        # Preprocessing
        "preprocessing_steps": [
            {"name": "crop_to_bbox", "enabled": False, "kwargs": {"bbox": [-10.0, 35.0, 30.0, 70.0]}}, # [min_lon, min_lat, max_lon, max_lat]
            {"name": "quality_filter", "enabled": True, "kwargs": {"threshold": 0.75}},
            {"name": "discard_small_files", "enabled": True, "kwargs": {"min_valid_points": 10}},
            {"name": "denoise_butterworth", "enabled": False, "kwargs": {"cutoff": 40, "order": 1}}, # # TODO not working properly currently
            {"name": "nan_gaussian_filter", "enabled": True, "kwargs": {"sigma": 1}},
            {"name": "local_standardization_m_mad", "enabled": True, "kwargs": {"window": 27}},
            {"name": "denoise_butterworth", "enabled": False, "kwargs": {"cutoff": 40, "order": 2}},
        ],

        # Detection: SCEA parameters
        "scea_parameters_algorithmic": {
            "growth_limit": 2,
            "detection_limit": 3.5,
            "max_pts_start_radius": 6,
            "local_box_size": 4,
            "metric": "geodesic",
            "point_value_threshold": "stds_from_median",
            "distance_matrix_kwargs": {"wind_height": "100m", "wind_scaling_method": "linear", "scaling_parameter": 0.04} # only used for "half_ellipse" metric, can include parameters for distance matrix calculation such as wind height and scaling parameter for wind influence on distance
        },
        "scea_parameters_runtime": {
            "row_cache_max_rows": 256,
            "verbose": False
        },

        # Results to save
        "output_fields": [
            {"name": "plume_id", "enabled": True, "kwargs": {"ignore_zero": True}}, # file_id + plume_number
            {"name": "plume_number", "enabled": True, "kwargs": {"ignore_zero": True}}, # unique within file, 0 is reserved for non-plume points
            {"name": "file_id", "enabled": False, "kwargs": {}},
            {"name": "file_name", "enabled": True, "kwargs": {}},
            {"name": "n_points_in_file", "enabled": True, "kwargs": {}},
            {"name": "max_point_indices", "enabled": True, "kwargs": {}},
            {"name": "timestamp_utc", "enabled": True, "kwargs": {}},
            {"name": "n_points", "enabled": True, "kwargs": {}},
            {"name": "max_point_locs", "enabled": True, "kwargs": {}},
            {"name": "max_point_value", "enabled": True, "kwargs": {}},
            {"name": "mean_point_value", "enabled": True, "kwargs": {}},
            {"name": "median_point_value", "enabled": True, "kwargs": {}},
            {"name": "bounding_box", "enabled": True, "kwargs": {}},
            {"name": "area", "enabled": False, "kwargs": {}},
            {"name": "wind_magnitude", "enabled": False, "kwargs": {}}, # TODO
            {"name": "wind_direction", "enabled": False, "kwargs": {}},
            {"name": "is_weekend", "enabled": False, "kwargs": {}},
            {"name": "day_of_week", "enabled": True, "kwargs": {}},
            {"name": "is_on_land", "enabled": True, "kwargs": {}},
            {"name": "median_of_neighbourhood", "enabled": True, "kwargs": {}},
            {"name": "is_max_point_on_edge", "enabled": True, "kwargs": {}},
            {"name": "is_plume_connected_to_nans_or_edge", "enabled": True, "kwargs": {}},
            {"name": "connected_plumes", "enabled": True, "kwargs": {}},
            {"name": "mean_q_value", "enabled": True, "kwargs": {}},
        ],

        # Runtime parameters
        "verbose": False,
        "logger_console_output": False,
        "logger_console_level": "DEBUG", # INFO or DEBUG
        "tqdm_enabled": True,
        "step_by_step": False, # if True, will give plots and prompt user to press Enter before each major step (preprocessing, SCEA, analysis) for debugging and inspection
        "step_by_step_plotting": {
            "open_in_browser": True,
            "max_background_points": 120000,
            "max_cluster_overlay_points": 120000,
            "max_cluster_markers": 4000,
            "cluster_info_marker_pixels": 1.0,
            "cluster_info_hover_pick_radius": 12,
            "skip_preprocessing_steps": ["discard_small_files"],
            "colorbar_tick_label_size": 4,
            "extent_pole_exclusion_degrees": 20.0,
            "render_mode": "pcolormesh_like",  # "points" or "pcolormesh_like"
            "rect_marker_size": 80,
            "map_background": True,
            "map_style": "carto-positron",  # open-street-map, carto-positron, carto-darkmatter
            "cartopy_resolution": "110m",
            "show_land": True,
            "show_ocean": True,
            "show_coastlines": True,
            "show_borders": False,
            "color_upper_quantile": 0.999,  # e.g. 0.95 or 0.99; set None to disable
            "color_lower_quantile": None,
            "value_cmap": "batlow",
            "cluster_cmap": "glasbey_light",
            "cluster_colorbar_max_ticks": 10,
            "figure_dpi": 150,
            "figure_width_per_col": 14.0,
            "figure_height": 9.5,
        },
    }


FIELD_ALIASES = {
    #"bounding_box": "bounding_boxes",
    "max_point_loc": "max_point_locs",
}

SPLIT_COLUMNS = {
    "max_point_indices": ["max_indx_row", "max_indx_col"],
    "max_point_locs": ["max_lon", "max_lat"],
    "bounding_box": ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"],
}
















# ============================
# CONFIG AND INPUT HANDLING
# ===========================

def load_config(config_file, logger=None) -> dict:
    if logger is None:
        logger = logging.getLogger("scea_plume_detection")
    
    if config_file is None:
        logger.info("No config file provided, using default configuration in script.")
        return default_config()
    else:
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_file}")

            if not isinstance(config, dict):
                logger.error("Loaded config must be a dictionary.")
                raise ValueError("Loaded config must be a dictionary.")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {str(e)}", exc_info=True)
            raise


def get_config_hash(config, config_file=None, length=5) -> str:
    # Canonical JSON hash so equivalent syntax (quotes/spacing/order) hashes the same
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]

def _snapshot_path(path, suffix=None):
    entries = []

    if suffix:
        suffix = suffix.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"

    if path is None:
        return entries

    if os.path.isfile(path):
        name = os.path.basename(path)
        if (suffix is None) or name.lower().endswith(suffix):
            st = os.stat(path)
            entries.append((name, st.st_size, st.st_mtime_ns))
        return entries

    if os.path.isdir(path):
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            if not os.path.isfile(fp):
                continue
            if suffix and not name.lower().endswith(suffix):
                continue
            st = os.stat(fp)
            entries.append((name, st.st_size, st.st_mtime_ns))
        entries.sort()
        return entries

    return entries

def get_input_hash(io_cfg, length=5) -> str:
    files_entries = _snapshot_path(
        io_cfg.get("files_dir"),
        suffix=io_cfg.get("files_type"),
    )

    wind_entries = _snapshot_path(
        io_cfg.get("wind_data_dir"),
        suffix=io_cfg.get("wind_data_type"),
    )

    payload_obj = {
        "files": files_entries,
        "wind": wind_entries,
    }

    payload = json.dumps(
        payload_obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:length]


def validate_config(config):
    # TODO implement function to validate config and raise errors if invalid

    # - if metric is "half_ellipse", check that wind data is provided and valid
    pass



    return data




def discover_input_files(io_cfg, logger=None) -> dict:
    if logger is None:
        logger = logging.getLogger("scea_plume_detection")
    
    files_path = sorted(glob(os.path.join(io_cfg["files_dir"], f"*.{io_cfg['files_type']}")))
    if io_cfg["file_indices_to_extract"] is not None:
        files_path = [files_path[i] for i in io_cfg["file_indices_to_extract"]]

    files_list = [os.path.basename(fp) for fp in files_path]

    files_time_range = None # TODO: implement time range extraction
    files_data_version = None # TODO: implement data version extraction

    logger.info(f"Discovered {len(files_path)} input files to process.")
    if files_path:
        logger.debug(f"Example file: {files_path[0]}")
    else:
        logger.warning("No input files found!")

    files_info = {
        "files_dir": io_cfg["files_dir"],
        "files_list": files_list,
        "files_path": files_path,
        "n_files": len(files_path),
        "files_time_range": files_time_range,
        "files_data_version": files_data_version,
        "first_file_example": files_path[0] if files_path else None
    }

    return files_info


def load_wind_data(io_cfg, scea_algo_cfg, logger=None, verbose=True) -> tuple:
    if logger is None:
        logger = logging.getLogger("scea_plume_detection")
    
    if "half_ellipse" in scea_algo_cfg["metric"]:
        # Get info about wind data
        # TODO add functionality to handle multiple wind files
        #wind_files_list = sorted(glob(os.path.join(io_cfg["wind_data_dir"], f"*.{io_cfg['wind_data_type']}")))
        wind_files_list = [io_cfg["wind_data_dir"]]
        wind_time_range = None # TODO: implement time range extraction for wind data
        wind_data_version = None # TODO: implement data version extraction for wind data

        wind_data = xr.open_dataset(
            wind_files_list[0],
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}
        )

        # Add wind direction and magnitude to the era5 dataset
        wind_angle_10m, wind_magnitude_10m = SCEA.wind_to_angle_and_magnitude(np.stack([wind_data.u10, wind_data.v10], axis=-1))
        wind_angle_100m, wind_magnitude_100m = SCEA.wind_to_angle_and_magnitude(np.stack([wind_data.u100, wind_data.v100], axis=-1))
        
        wind_data['wind_angle_10m'] = (('time','latitude', 'longitude'), wind_angle_10m)
        wind_data['wind_magnitude_10m'] = (('time', 'latitude', 'longitude',), wind_magnitude_10m)
        wind_data['wind_angle_100m'] = (('time','latitude', 'longitude'), wind_angle_100m)
        wind_data['wind_magnitude_100m'] = (('time', 'latitude', 'longitude',), wind_magnitude_100m)

        if verbose:
            print(f"[Init] Found {len(wind_files_list)} wind data files.")
            print(f"[Init] Time range of wind data files: {wind_time_range}")
            print(f"[Init] Data version of wind data files: {wind_data_version}\n")

        wind_height = scea_algo_cfg["distance_matrix_kwargs"]["wind_height"]
        #if wind_height not in ["10m", "100m"]:
        #    raise ValueError(f"Invalid wind height specified in config: {wind_height}. Must be '10m' or '100m' to match available wind data variables.")
        
        wind_scaling_method = scea_algo_cfg["distance_matrix_kwargs"]["wind_scaling_method"]
        wind_scaling_parameter = scea_algo_cfg["distance_matrix_kwargs"]["scaling_parameter"]
        

    else:
        wind_data = None
        wind_files_list = None
        wind_time_range = None
        wind_data_version = None
        wind_height = None
        wind_scaling_method = None
        wind_scaling_parameter = None

    wind_data_info = {
        "wind_files_list": wind_files_list,
        "wind_time_range": wind_time_range,
        "wind_data_version": wind_data_version,
        "wind_height": wind_height,
        "wind_scaling_method": wind_scaling_method,
        "wind_scaling_parameter": wind_scaling_parameter
    }

    return wind_data, wind_data_info


def collect_run_metadata(config_file, config, config_hash, input_hash, files_info, wind_data_info, code_version=None, verbose=True, logger=None) -> dict:

    timestamp_local = datetime.datetime.now()
    run_id = f"run_{timestamp_local.strftime('%y%m%d_%H%M%S')}_C{config_hash}_I{input_hash}"

    if logger is None:
        logger = logging.getLogger("scea_plume_detection")

    logger.info(f"\n[Init] Starting SCEA run with ID: {run_id}")
    if verbose:
        logger.info(f"[Init] Timestamp (Local): {timestamp_local.isoformat()}")
        logger.info(f"[Init] Code version: {code_version}\n")

    metadata = {
        "run_id": run_id,
        "timestamp_local": timestamp_local.isoformat(),
        "config_hash": config_hash,
        "input_hash": input_hash,
        "code_version": code_version,
        "config_file": config_file,
        "config": copy.deepcopy(config),
        "files_info": files_info,
        "wind_data_info": wind_data_info
    }

    return metadata



# =============================
# LOGGING SETUP
# =============================

def parse_log_level(level_str):
    """Convert string to logging level constant."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str.upper(), logging.INFO)

def setup_logging(log_dir, run_id, level=logging.INFO, console_output=True):
    """
    Configure logging with file and optional console handlers.
    
    Args:
        log_dir: Directory to store log files
        run_id: Run identifier for log file naming
        level: Logging level (default: INFO)
        console_output: Whether to print to terminal (default: True)
    
    Returns:
        logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("scea_plume_detection")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    log_file = os.path.join(log_dir, f"scea_run_{run_id}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Only add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger



# ============================
# PREPROSESSING
# ============================

def _format_kwargs_for_title(kwargs):
    if not kwargs:
        return "{}"
    try:
        return json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(kwargs)


def _values_2d(da_or_array):
    arr = np.ma.asarray(da_or_array)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr[0]
    return np.asarray(arr)


def _coerce_numeric_1d(arr):
    out = pd.to_numeric(np.asarray(arr).reshape(-1), errors="coerce")
    return np.asarray(out, dtype=float)


def _downsample_flat(lon, lat, values, max_points):
    n = len(lon)
    if max_points is None or n <= max_points:
        return lon, lat, values
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
    return lon[idx], lat[idx], values[idx]


def _normalize_quantile_param(value):
    if value is None:
        return None
    q = float(value)
    if q > 1.0:
        q = q / 100.0
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0
    return q


def _compute_color_limits(values, plotting_cfg):
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None

    q_low = _normalize_quantile_param(plotting_cfg.get("color_lower_quantile", None))
    q_high = _normalize_quantile_param(plotting_cfg.get("color_upper_quantile", None))

    cmin = np.nanmin(arr) if q_low is None else np.nanquantile(arr, q_low)
    cmax = np.nanmax(arr) if q_high is None else np.nanquantile(arr, q_high)

    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
        return None, None
    return float(cmin), float(cmax)


def _quality_filter_threshold(preprocessing_steps_cfg):
    for step_cfg in preprocessing_steps_cfg or []:
        if step_cfg.get("name") == "quality_filter" and step_cfg.get("enabled", False):
            threshold = step_cfg.get("kwargs", {}).get("threshold", None)
            if threshold is None:
                return None
            try:
                return float(threshold)
            except Exception:
                return None
    return None


def _should_skip_preprocessing_step_plot(step_name, plotting_cfg):
    skip_steps = plotting_cfg.get("skip_preprocessing_steps", []) or []
    return step_name in set(skip_steps)


def _continuous_value_cmap(plotting_cfg):
    cmap_name = plotting_cfg.get("value_cmap", "batlow")
    if cmap_name == "batlow" and cmc is not None:
        return cmc.batlow
    try:
        return plt.get_cmap(cmap_name)
    except Exception:
        if cmc is not None:
            return cmc.batlow
        return plt.get_cmap("viridis")


def _dataset_to_plot_arrays(data, variable_names_cfg, quality_threshold=None):
    lon = np.ma.asarray(data[variable_names_cfg["lon"]])
    lat = np.ma.asarray(data[variable_names_cfg["lat"]])
    values = np.ma.asarray(data[variable_names_cfg["value"]])

    if quality_threshold is not None and variable_names_cfg.get("quality") in data:
        quality = np.ma.asarray(data[variable_names_cfg["quality"]])
        quality = quality.filled(np.nan) if np.ma.isMaskedArray(quality) else np.asarray(quality)
        values = np.ma.asarray(values, dtype=float)
        values = np.where(np.asarray(quality, dtype=float) >= float(quality_threshold), values, np.nan)

    if np.ma.isMaskedArray(lon):
        lon = lon.filled(np.nan)
    if np.ma.isMaskedArray(lat):
        lat = lat.filled(np.nan)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    values = _values_2d(values)
    return lon, lat, values


def _flatten_clusters_like_values(clusters):
    return _coerce_numeric_1d(_values_2d(clusters))


def _cluster_grid_for_plot(clusters):
    cluster_grid = np.asarray(_values_2d(clusters), dtype=float)
    # Explicitly match notebook behavior: cluster label 0 should not be plotted.
    cluster_grid = np.where(cluster_grid == 0, np.nan, cluster_grid)
    cluster_grid[~np.isfinite(cluster_grid)] = np.nan
    cluster_grid[cluster_grid < 0] = np.nan
    return cluster_grid


def _categorical_cluster_cmap(max_cluster_id, plotting_cfg):
    base_name = plotting_cfg.get("cluster_cmap", "glasbey_light")
    max_cluster_id = int(max(1, max_cluster_id))

    base = None
    if base_name == "glasbey_light" and cc is not None:
        base = cc.cm.glasbey_light
    elif base_name == "glasbey" and cc is not None:
        base = cc.cm.glasbey
    else:
        try:
            base = plt.get_cmap(base_name, max_cluster_id)
        except Exception:
            base = plt.get_cmap("tab20", max_cluster_id)

    if hasattr(base, "colors"):
        colors = np.asarray(base.colors)
        if colors.shape[0] < max_cluster_id:
            reps = int(np.ceil(max_cluster_id / colors.shape[0]))
            colors = np.tile(colors, (reps, 1))
        colors = colors[:max_cluster_id]
    else:
        colors = base(np.linspace(0, 1, max_cluster_id))

    cmap = ListedColormap(colors, name=f"{base_name}_{max_cluster_id}")
    cmap.set_bad((0, 0, 0, 0))
    boundaries = np.arange(0.5, max_cluster_id + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _data_extent(lon, lat, values=None, plotting_cfg=None):
    lon = np.asarray(lon).reshape(-1)
    lat = np.asarray(lat).reshape(-1)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if values is not None:
        values = np.asarray(values).reshape(-1)
        finite = finite & np.isfinite(values)
    lon = lon[finite]
    lat = lat[finite]
    if lon.size == 0 or lat.size == 0:
        return (-180.0, 180.0, -90.0, 90.0)

    pole_exclusion_degrees = 0.0
    if plotting_cfg is not None:
        pole_exclusion_degrees = float(plotting_cfg.get("extent_pole_exclusion_degrees", 0.0) or 0.0)
        if pole_exclusion_degrees < 0.0:
            pole_exclusion_degrees = 0.0
        if pole_exclusion_degrees > 90.0:
            pole_exclusion_degrees = 90.0

    if pole_exclusion_degrees > 0:
        lat_min_allowed = -90.0 + pole_exclusion_degrees
        lat_max_allowed = 90.0 - pole_exclusion_degrees
        core = (lat >= lat_min_allowed) & (lat <= lat_max_allowed)
        if np.count_nonzero(core) >= 10:
            lon_core = lon[core]
            lat_core = lat[core]
        else:
            lon_core = lon
            lat_core = lat
    else:
        lon_core = lon
        lat_core = lat

    lon_min, lon_max = float(np.min(lon_core)), float(np.max(lon_core))
    lat_min, lat_max = float(np.min(lat_core)), float(np.max(lat_core))
    lon_pad = max((lon_max - lon_min) * 0.03, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.03, 0.1)
    return (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)


def _setup_debug_axes(n_cols, plotting_cfg, extent):
    use_cartopy = ccrs is not None and cfeature is not None and plotting_cfg.get("map_background", True)
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
    fig_width = plotting_cfg.get("figure_width_per_col", 8.0) * n_cols
    fig_height = plotting_cfg.get("figure_height", 6.5)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), squeeze=False, subplot_kw=subplot_kw, constrained_layout=True)
    axes = list(axes[0])
    fig.set_dpi(plotting_cfg.get("figure_dpi", 100))

    for ax in axes:
        if use_cartopy:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            resolution = plotting_cfg.get("cartopy_resolution", "110m")
            if plotting_cfg.get("show_land", True):
                ax.add_feature(cfeature.LAND.with_scale(resolution), facecolor="#f3efe8", zorder=0)
            if plotting_cfg.get("show_ocean", True):
                ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor="#dceefb", zorder=0)
            if plotting_cfg.get("show_coastlines", True):
                ax.coastlines(resolution=resolution, linewidth=0.45, color="#444444", zorder=1)
            if plotting_cfg.get("show_borders", False):
                ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.25, edgecolor="#666666", zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.25, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

    return fig, axes, use_cartopy


def _pcolormesh_panel(ax, lon, lat, values, plotting_cfg, title, show_colorbar=False, cmap="batlow", alpha=1.0, use_cartopy=False, norm=None):
    values = np.ma.asarray(values, dtype=float)
    values = np.ma.masked_invalid(values)
    if norm is None:
        cmin, cmax = _compute_color_limits(values, plotting_cfg)
        norm = Normalize(vmin=cmin, vmax=cmax) if (Normalize is not None and cmin is not None and cmax is not None) else None
    kwargs = dict(cmap=cmap, shading="auto", alpha=alpha, norm=norm, edgecolors="none", antialiased=False, rasterized=True)
    if use_cartopy:
        kwargs["transform"] = ccrs.PlateCarree()
    m = ax.pcolormesh(lon, lat, values, **kwargs)
    ax.set_title(title)
    if show_colorbar:
        return m
    return m


def _add_colorbar(fig, mappable, ax, plotting_cfg, **kwargs):
    cbar = fig.colorbar(mappable, ax=ax, **kwargs)
    label_size = plotting_cfg.get("colorbar_tick_label_size", 5)
    cbar.ax.tick_params(labelsize=label_size)
    return cbar


def _add_cluster_scatter(ax, lon, lat, color_values, use_cartopy=False, cmap="turbo", title=None):
    scatter_kwargs = dict(c=color_values, cmap=cmap, s=16, edgecolors="black", linewidths=0.2)
    if use_cartopy:
        scatter_kwargs["transform"] = ccrs.PlateCarree()
    sc = ax.scatter(lon, lat, **scatter_kwargs)
    if title:
        ax.set_title(title)
    return sc


def _save_and_show_figure(fig, out_path, logger=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_dpi = 120
    try:
        save_dpi = fig.get_dpi()
    except Exception:
        pass
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    if logger is not None:
        logger.info(f"Saved debug plot: {out_path}")
    try:
        fig.canvas.manager.set_window_title(os.path.basename(out_path))
    except Exception:
        pass
    plt.show(block=False)
    try:
        fig.canvas.draw_idle()
        plt.pause(0.05)
    except Exception:
        pass


def _pause_for_step(message, fig=None):
    try:
        input(message)
    except EOFError:
        pass
    if fig is not None:
        plt.close(fig)
    plt.close("all")
    gc.collect()


def plot_preprocessing_step_debug(
    original_data,
    before_data,
    after_data,
    variable_names_cfg,
    original_quality_threshold,
    step_name,
    step_kwargs,
    debug_dir,
    run_id,
    file_id,
    step_index,
    plotting_cfg,
    logger=None,
):
    if plt is None:
        if logger is not None:
            logger.warning("Matplotlib is unavailable. Skipping debug plot.")
        return

    original_lon, original_lat, original_val = _dataset_to_plot_arrays(original_data, variable_names_cfg, quality_threshold=original_quality_threshold)
    before_lon, before_lat, before_val = _dataset_to_plot_arrays(before_data, variable_names_cfg)
    after_lon, after_lat, after_val = _dataset_to_plot_arrays(after_data, variable_names_cfg)

    panels = [
        ("original", original_lon, original_lat, original_val),
        ("before", before_lon, before_lat, before_val),
        ("after", after_lon, after_lat, after_val),
    ]

    extent = _data_extent(
        np.concatenate([_coerce_numeric_1d(p[1]) for p in panels]),
        np.concatenate([_coerce_numeric_1d(p[2]) for p in panels]),
        np.concatenate([_coerce_numeric_1d(p[3]) for p in panels]),
        plotting_cfg,
    )
    fig, axes, use_cartopy = _setup_debug_axes(len(panels), plotting_cfg, extent)

    kwargs_text = _format_kwargs_for_title(step_kwargs)
    fig.suptitle(
        f"run={run_id} | file={file_id} | preprocessing step {step_index + 1}: {step_name}\nkwargs={kwargs_text}",
        fontsize=11,
    )

    for ax, (title, lon, lat, values) in zip(axes, panels, strict=False):
        m = _pcolormesh_panel(ax, lon, lat, values.data, plotting_cfg, title, show_colorbar=False, cmap=_continuous_value_cmap(plotting_cfg), alpha=1.0, use_cartopy=use_cartopy)
        _add_colorbar(fig, m, ax, plotting_cfg, shrink=0.34, pad=0.02, fraction=0.02)

    out_path = os.path.join(debug_dir, f"{run_id}_{file_id}_preprocess_{step_index + 1:02d}_{step_name}.png")
    _save_and_show_figure(fig, out_path, logger=logger)
    _pause_for_step(f"[step-by-step] Inspect preprocessing step '{step_name}'. Press Enter to continue...", fig=fig)


def plot_clustering_debug(
    original_data,
    preprocessed_data,
    clusters,
    variable_names_cfg,
    original_quality_threshold,
    debug_dir,
    run_id,
    file_id,
    plotting_cfg,
    logger=None,
):
    if plt is None:
        if logger is not None:
            logger.warning("Matplotlib is unavailable. Skipping debug plot.")
        return

    max_cluster_points = plotting_cfg.get("max_cluster_overlay_points", 120000)

    o_lon, o_lat, o_val = _dataset_to_plot_arrays(original_data, variable_names_cfg, quality_threshold=original_quality_threshold)
    p_lon, p_lat, p_val = _dataset_to_plot_arrays(preprocessed_data, variable_names_cfg)
    p_lon_flat = _coerce_numeric_1d(p_lon)
    p_lat_flat = _coerce_numeric_1d(p_lat)
    p_val_flat = _coerce_numeric_1d(p_val)

    cluster_grid = _cluster_grid_for_plot(clusters)

    grid_rows = min(np.asarray(o_val).shape[0], np.asarray(p_val).shape[0], cluster_grid.shape[0])
    grid_cols = min(np.asarray(o_val).shape[1], np.asarray(p_val).shape[1], cluster_grid.shape[1])
    o_lon = np.asarray(o_lon)[:grid_rows, :grid_cols]
    o_lat = np.asarray(o_lat)[:grid_rows, :grid_cols]
    o_val = np.asarray(o_val)[:grid_rows, :grid_cols]
    p_lon = np.asarray(p_lon)[:grid_rows, :grid_cols]
    p_lat = np.asarray(p_lat)[:grid_rows, :grid_cols]
    p_val = np.asarray(p_val)[:grid_rows, :grid_cols]
    cluster_grid = np.asarray(cluster_grid)[:grid_rows, :grid_cols]
    cluster_ids = cluster_grid[np.isfinite(cluster_grid)]
    max_cluster_id = int(np.nanmax(cluster_ids)) if cluster_ids.size else 0
    cluster_cmap, cluster_norm = _categorical_cluster_cmap(max_cluster_id, plotting_cfg) if max_cluster_id > 0 else (None, None)

    panels = [
        ("original", o_lon, o_lat, o_val),
        ("preprocessed", p_lon, p_lat, p_val),
        ("preprocessed + clusters", p_lon, p_lat, p_val),
    ]
    extent = _data_extent(
        np.concatenate([_coerce_numeric_1d(p[1]) for p in panels]),
        np.concatenate([_coerce_numeric_1d(p[2]) for p in panels]),
        np.concatenate([_coerce_numeric_1d(p[3]) for p in panels]),
        plotting_cfg,
    )
    fig, axes, use_cartopy = _setup_debug_axes(3, plotting_cfg, extent)

    fig.suptitle(f"run={run_id} | file={file_id} | clustering result", fontsize=11)
    value_cmap = _continuous_value_cmap(plotting_cfg)
    bg1 = _pcolormesh_panel(axes[0], o_lon, o_lat, o_val, plotting_cfg, "original", show_colorbar=False, cmap=value_cmap, alpha=1.0, use_cartopy=use_cartopy)
    bg2 = _pcolormesh_panel(axes[1], p_lon, p_lat, p_val, plotting_cfg, "preprocessed", show_colorbar=False, cmap=value_cmap, alpha=1.0, use_cartopy=use_cartopy)
    bg3 = _pcolormesh_panel(axes[2], p_lon, p_lat, p_val, plotting_cfg, "preprocessed + clusters", show_colorbar=False, cmap=value_cmap, alpha=0.9, use_cartopy=use_cartopy)
    _add_colorbar(fig, bg1, axes[0], plotting_cfg, shrink=0.34, pad=0.02, fraction=0.02)
    _add_colorbar(fig, bg2, axes[1], plotting_cfg, shrink=0.34, pad=0.02, fraction=0.02)
    _add_colorbar(fig, bg3, axes[2], plotting_cfg, shrink=0.34, pad=0.02, fraction=0.02)
    if max_cluster_id > 0:
        cluster_mesh = _pcolormesh_panel(
            axes[2],
            p_lon,
            p_lat,
            cluster_grid,
            plotting_cfg,
            "preprocessed + clusters",
            show_colorbar=False,
            cmap=cluster_cmap,
            alpha=0.92,
            use_cartopy=use_cartopy,
            norm=cluster_norm,
        )
        tick_count = max(2, int(plotting_cfg.get("cluster_colorbar_max_ticks", 10)))
        tick_step = max(1, max_cluster_id // tick_count)
        ticks = np.arange(1, max_cluster_id + 1, tick_step)
    else:
        axes[2].text(0.5, 0.5, "No clusters detected", transform=axes[2].transAxes, ha="center", va="center")

    out_path = os.path.join(debug_dir, f"{run_id}_{file_id}_clustering.png")
    _save_and_show_figure(fig, out_path, logger=logger)
    _pause_for_step("[step-by-step] Inspect clustering map. Press Enter to continue...", fig=fig)


def _build_cluster_hover_text(row):
    parts = []
    for col in row.index:
        val = row[col]
        if isinstance(val, float):
            val_str = f"{val:.6g}"
        else:
            val_str = str(val)
        parts.append(f"{col}: {val_str}")
    return "\n".join(parts)


def plot_cluster_outputs_debug(
    preprocessed_data,
    clusters,
    cluster_results,
    variable_names_cfg,
    debug_dir,
    run_id,
    file_id,
    plotting_cfg,
    logger=None,
):
    if plt is None:
        if logger is not None:
            logger.warning("Matplotlib is unavailable. Skipping debug plot.")
        return

    max_cluster_markers = plotting_cfg.get("max_cluster_markers", 4000)

    p_lon, p_lat, p_val = _dataset_to_plot_arrays(preprocessed_data, variable_names_cfg)
    p_lon_flat = _coerce_numeric_1d(p_lon)
    p_lat_flat = _coerce_numeric_1d(p_lat)
    p_val_flat = _coerce_numeric_1d(p_val)

    cluster_grid = _cluster_grid_for_plot(clusters)

    grid_rows = min(np.asarray(p_val).shape[0], cluster_grid.shape[0])
    grid_cols = min(np.asarray(p_val).shape[1], cluster_grid.shape[1])
    p_lon = np.asarray(p_lon)[:grid_rows, :grid_cols]
    p_lat = np.asarray(p_lat)[:grid_rows, :grid_cols]
    p_val = np.asarray(p_val)[:grid_rows, :grid_cols]
    cluster_grid = np.asarray(cluster_grid)[:grid_rows, :grid_cols]

    cluster_ids = cluster_grid[np.isfinite(cluster_grid)]
    max_cluster_id = int(np.nanmax(cluster_ids)) if cluster_ids.size else 0
    cluster_cmap, cluster_norm = _categorical_cluster_cmap(max_cluster_id, plotting_cfg) if max_cluster_id > 0 else (None, None)

    if cluster_results is None or cluster_results.empty:
        if logger is not None:
            logger.info("No cluster output rows to visualize in step-by-step output stage.")
        return

    if not {"max_lon", "max_lat"}.issubset(cluster_results.columns):
        if logger is not None:
            logger.warning("Cluster output map skipped: max_lon/max_lat columns are missing.")
        return

    marker_df = cluster_results.copy()
    marker_df = marker_df[np.isfinite(marker_df["max_lon"]) & np.isfinite(marker_df["max_lat"])].copy()
    if len(marker_df) > max_cluster_markers:
        marker_df = marker_df.sample(n=max_cluster_markers, random_state=0)

    marker_rows = marker_df.reset_index(drop=True)
    marker_texts = marker_rows.apply(_build_cluster_hover_text, axis=1).to_list()

    extent = _data_extent(p_lon_flat, p_lat_flat, p_val_flat, plotting_cfg)
    fig, axes, use_cartopy = _setup_debug_axes(1, plotting_cfg, extent)
    ax = axes[0]
    fig.suptitle(
        f"run={run_id} | file={file_id} | clustering + plume output metadata (click markers)",
        fontsize=11,
    )

    value_cmap = _continuous_value_cmap(plotting_cfg)
    bg = _pcolormesh_panel(ax, p_lon, p_lat, p_val, plotting_cfg, "preprocessed + clusters + plume info points", show_colorbar=False, cmap=value_cmap, alpha=0.9, use_cartopy=use_cartopy)
    if max_cluster_id > 0:
        cluster_mesh = _pcolormesh_panel(
            ax,
            p_lon,
            p_lat,
            cluster_grid,
            plotting_cfg,
            "preprocessed + clusters + plume info points",
            show_colorbar=False,
            cmap=cluster_cmap,
            alpha=0.92,
            use_cartopy=use_cartopy,
            norm=cluster_norm,
        )
        tick_count = max(2, int(plotting_cfg.get("cluster_colorbar_max_ticks", 10)))
        tick_step = max(1, max_cluster_id // tick_count)
        ticks = np.arange(1, max_cluster_id + 1, tick_step)
    else:
        ax.text(0.5, 0.5, "No clusters detected", transform=ax.transAxes, ha="center", va="center")

    marker_grid = np.ma.masked_all((grid_rows, grid_cols), dtype=float)
    marker_texts_by_cell = {}
    marker_lon_values = []
    marker_lat_values = []

    max_indx_row_col = {"max_indx_row", "max_indx_col"}.issubset(marker_rows.columns)
    if max_indx_row_col:
        marker_rows["max_indx_row"] = pd.to_numeric(marker_rows["max_indx_row"], errors="coerce")
        marker_rows["max_indx_col"] = pd.to_numeric(marker_rows["max_indx_col"], errors="coerce")

    for idx, r in marker_rows.iterrows():
        row_idx = None
        col_idx = None
        if max_indx_row_col:
            row_val = r["max_indx_row"]
            col_val = r["max_indx_col"]
            if np.isfinite(row_val) and np.isfinite(col_val):
                row_idx = int(row_val)
                col_idx = int(col_val)
                if not (0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols):
                    row_idx = None
                    col_idx = None

        if row_idx is None or col_idx is None:
            try:
                cx = float(r["max_lon"])
                cy = float(r["max_lat"])
                dist2 = (np.asarray(p_lon, dtype=float) - cx) ** 2 + (np.asarray(p_lat, dtype=float) - cy) ** 2
                flat_idx = int(np.nanargmin(dist2))
                row_idx, col_idx = np.unravel_index(flat_idx, dist2.shape)
            except Exception:
                continue

        marker_grid[row_idx, col_idx] = 1.0
        marker_texts_by_cell[(row_idx, col_idx)] = marker_texts[idx]
        marker_lon_values.append(float(p_lon[row_idx, col_idx]))
        marker_lat_values.append(float(p_lat[row_idx, col_idx]))

    marker_cmap = ListedColormap(["#101010"])
    marker_cmap.set_bad((0, 0, 0, 0))
    marker_norm = Normalize(vmin=0.5, vmax=1.5) if Normalize is not None else None
    marker_mesh = _pcolormesh_panel(
        ax,
        p_lon,
        p_lat,
        marker_grid,
        plotting_cfg,
        "preprocessed + clusters + plume info points",
        show_colorbar=False,
        cmap=marker_cmap,
        alpha=0.95,
        use_cartopy=use_cartopy,
        norm=marker_norm,
    )

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ann.set_visible(False)

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if not marker_lon_values:
            return

        click_scale = float(plotting_cfg.get("cluster_info_click_hit_scale", 1.5) or 1.5)
        try:
            lon_diffs = np.abs(np.diff(np.asarray(p_lon, dtype=float), axis=1))
            lon_res = float(np.nanmedian(lon_diffs)) if lon_diffs.size else 0.01
        except Exception:
            lon_res = 0.01
        try:
            lat_diffs = np.abs(np.diff(np.asarray(p_lat, dtype=float), axis=0))
            lat_res = float(np.nanmedian(lat_diffs)) if lat_diffs.size else 0.01
        except Exception:
            lat_res = 0.01

        lon_tol = max(lon_res * click_scale, 1e-12)
        lat_tol = max(lat_res * click_scale, 1e-12)

        marker_lon_arr = np.asarray(marker_lon_values, dtype=float)
        marker_lat_arr = np.asarray(marker_lat_values, dtype=float)
        norm_dx = (marker_lon_arr - float(event.xdata)) / lon_tol
        norm_dy = (marker_lat_arr - float(event.ydata)) / lat_tol
        dist2 = norm_dx**2 + norm_dy**2
        idx = int(np.argmin(dist2))

        if not np.isfinite(dist2[idx]) or dist2[idx] > 1.0:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return

        row = marker_rows.iloc[idx]
        cell_key = None
        if max_indx_row_col and np.isfinite(row["max_indx_row"]) and np.isfinite(row["max_indx_col"]):
            cell_key = (int(row["max_indx_row"]), int(row["max_indx_col"]))

        if cell_key in marker_texts_by_cell:
            ann.set_text(marker_texts_by_cell[cell_key])
        else:
            ann.set_text(marker_texts[idx])
        ann.xy = (float(marker_lon_arr[idx]), float(marker_lat_arr[idx]))
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)

    out_path = os.path.join(debug_dir, f"{run_id}_{file_id}_cluster_outputs_click.png")
    _save_and_show_figure(fig, out_path, logger=logger)
    _pause_for_step("[step-by-step] Inspect output map. Click a max-point marker to see plume metadata, then press Enter to continue...", fig=fig)

def normalize_singleton_leading_dim(data, variable_names_cfg, logger=None, verbose=True) -> xr.Dataset:
    if logger is None:
        logger = logging.getLogger("scea_plume_detection")
    """
    Normalize value field shape from [1, x, y] -> [x, y].
    If leading dim is >1, fail fast to avoid silently dropping data.
    """
    value_var = variable_names_cfg["value"]
    da = data[value_var]

    if da.ndim <= 2:
        return data

    lead_dim = da.dims[0]
    lead_size = da.sizes[lead_dim]

    if lead_size == 1:
        if verbose: print(f"Squeezing singleton leading dimension '{lead_dim}' for '{value_var}'.")
        return data.isel({lead_dim: 0})

    raise ValueError(
    f"Expected singleton leading dimension for '{value_var}', "
    f"but got {lead_dim} size {lead_size} with shape {da.shape}."
    )


def preprocess_for_scea(
    data,
    preprocessing_steps_cfg,
    variable_names_cfg,
    logger=None,
    verbose=True,
    tqdm=None,
    step_by_step=False,
    original_data=None,
    on_step_complete=None,
) -> xr.Dataset:
    """
    data: xarray dataset containing the data to preprocess
    """
    if logger is None:
        logger = logging.getLogger("scea_plume_detection")

    if tqdm is not None:
        preprocessing_steps_print_list = ["⏳"+step_cfg["name"][:11] for step_cfg in preprocessing_steps_cfg if step_cfg["enabled"]]
        status = f"  [0]Preprocessing ({', '.join(preprocessing_steps_print_list)})" if preprocessing_steps_print_list else "Preprocessing"
        tqdm.set_description(status)

    j=0
    for step_idx, step_cfg in enumerate(preprocessing_steps_cfg):
        if step_cfg["enabled"]:

            step_kwargs = step_cfg.get("kwargs", {}) # get kwargs for this step, or empty dict if not provided

            logger.debug(f"Preprocessing: {step_cfg['name']} with kwargs: {step_kwargs}")

            data_before = data.copy(deep=True) if step_by_step else None

            data = SCEA.preprocess_data_xr(data, step_cfg["name"], step_kwargs, variable_names_cfg, verbose=verbose)

            if step_by_step and on_step_complete is not None and data is not None:
                on_step_complete(
                    step_idx=step_idx,
                    step_cfg=step_cfg,
                    before_data=data_before,
                    after_data=data,
                    original_data=original_data,
                )

            if tqdm is not None:
                preprocessing_steps_print_list[j] = "✅" + preprocessing_steps_print_list[j][1:]
                tqdm.set_description(f"  [{j+1}] Preprocessing ({', '.join(preprocessing_steps_print_list)})")
                #tqdm.set_postfix({"step": step_cfg["name"]})
                j += 1


            if data is None:
                logger.warning(f"Data discarded after preprocessing step: {step_cfg['name']}. Skipping file.")
                return None
        
        #if tqdm is not None:
        #    tqdm.update(1)
    
    #if tqdm is not None:
    #    tqdm.set_description(f"Finished preprocessing")

    return data


def get_max_point_indices_array(results: pd.DataFrame):
    """
    Return numeric (n,2) array for SCEA internal indexing.
    Prefers split numeric columns; falls back to old object column if present.
    """
    if {"max_row", "max_col"}.issubset(results.columns):
        return results[["max_row", "max_col"]].to_numpy(dtype=int)

    if "max_point_indices" in results.columns:
        s = results["max_point_indices"]
        if s.notna().any():
            return pd.DataFrame(s.tolist(), index=results.index).to_numpy(dtype=int)

    return None


# ========================================
# CLUSTER ANALYSIS AND OUTPUT PROCESSING
# ========================================

def build_output_columns(output_config):
    columns = []

    for field in output_config:
        if not field["enabled"]:
            continue

        field_name = field["name"]
        method_name = FIELD_ALIASES.get(field_name, field_name)

        if method_name in SPLIT_COLUMNS:
            columns.extend(SPLIT_COLUMNS[method_name])
        else:
            columns.append(field_name)

    # Add file metadata if you always append these later
    for extra in ["file_id", "file_name"]:
        if extra not in columns:
            columns.append(extra)

    return columns


def assign_cluster_output(results: pd.DataFrame, field_name: str, method_name, raw, index):
    """
    Write raw output from SCEA into results:
    - scalar -> column with repeated scalar
    - 1D -> one column
    - 2D fixed-width -> split into numeric columns
    - fallback -> object column
    """

    # split into columns
    if method_name in SPLIT_COLUMNS:
        cols = SPLIT_COLUMNS[method_name]
        arr = np.asarray(raw)

        if arr.ndim != 2 or arr.shape[1] != len(cols):
            raise ValueError(f"{method_name} expected shape (n, {len(cols)}), got {arr.shape}")

        results[cols] = pd.DataFrame(arr, index=index, columns=cols)
        return


    # Keep list/set-like variable-length outputs as object
    if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (set, list, tuple, dict)):
        results[field_name] = pd.Series(list(raw), index=index, dtype=object)
        return

    arr = np.asarray(raw)

    # Scalar
    if arr.ndim == 0:
        results[field_name] = pd.Series([arr.item()] * len(index), index=index)
        return

    # 1D vector
    if arr.ndim == 1:
        results[field_name] = pd.Series(arr, index=index)
        return

    # Fallback
    results[field_name] = pd.Series(list(raw), index=index, dtype=object)


def normalize_output_for_dataframe(value, index=None):
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == 2:
            return pd.Series([list(row) for row in value], index=index, dtype=object)
        if value.ndim == 1:
            return pd.Series(value, index=index)
        return pd.Series(list(value), index=index, dtype=object)

    if isinstance(value, (list, tuple)):
        return pd.Series(value, index=index, dtype=object)

    return value


def analyze_scea_clusters(clusters, output_config, lon=None, lat=None, values=None, timestamps=None, qa_values=None, variable_names=None, ignore_zero=True, logger=None, verbose=True, filename=None, tqdm=None) -> list:
    """
    returns a pandas dataframe with one row per cluster and columns specified in output_config
    """
    if logger is None:
        logger = logging.getLogger("scea_plume_detection")

    columns = build_output_columns(output_config)
    n_clusters = int(np.nanmax(clusters))
    
    if ignore_zero:
        index = range(1, n_clusters + 1)
    else:
        index = range(0, n_clusters + 1)
    
    logger.info(f"Analyzing {n_clusters} SCEA clusters.")


    # Initialize results dataframe
    results = pd.DataFrame(index=index, columns=columns)


    if n_clusters == 0:
        logger.warning("No clusters detected, skipping analysis.")
        return results.reset_index(drop=True)


    tqdm_fields = [field_cfg for field_cfg in output_config if field_cfg["enabled"]]
    if tqdm is not None:
        tqdm_fields_print = ["⏳" +field_cfg["name"][:2] for field_cfg in tqdm_fields]
        tqdm.set_description(f"  [0]Analyzing {','.join(tqdm_fields_print)}")

    # Pre-compute max_point_indices first (regardless of config position)
    # so it's available for other methods that may depend on it
    max_point_indices = None
    for output_field in output_config:
        if not output_field["enabled"]:
            continue
        field_name = output_field["name"]
        method_name = FIELD_ALIASES.get(field_name, field_name)
        if method_name == "max_point_indices":

            step_kwargs = output_field.get("kwargs", {})
            max_point_indices = np.asarray(SCEA.analyze_clusters(
                method=method_name,
                clusters=clusters,
                lon=lon,
                lat=lat,
                values=values,
                timestamps=timestamps,
                qa_values=qa_values,
                variable_names=variable_names,
                max_point_indices=None,
                ignore_zero=ignore_zero,
                verbose=verbose,
                filename=filename,
                kwargs=step_kwargs,
            ))

            if tqdm is not None:
                tqdm_fields_print[tqdm_fields.index(output_field)] = "✅" + tqdm_fields_print[tqdm_fields.index(output_field)][1:]
                tqdm.set_description(f"  [0]Analyzing {','.join(tqdm_fields_print)}")

            break

    # Now process all output fields (max_point_indices already cached)
    j=0
    for output_field in output_config:
        if not output_field["enabled"]:
            continue

        field_name = output_field["name"]
        method_name = FIELD_ALIASES.get(field_name, field_name)
        step_kwargs = output_field.get("kwargs", {})
        
        raw = SCEA.analyze_clusters(
            method=method_name,
            clusters=clusters,
            lon=lon,
            lat=lat,
            values=values,
            timestamps=timestamps,
            qa_values=qa_values,
            variable_names=variable_names,
            max_point_indices=max_point_indices,
            ignore_zero=ignore_zero,
            verbose=verbose,
            filename=filename,
            kwargs=step_kwargs,
        )

        assign_cluster_output(results, field_name, method_name, raw, results.index)
        
        if tqdm is not None:

            tqdm_fields_print[j] = "✅" + tqdm_fields_print[j][1:]
            tqdm.set_description(f"  [{j+1}]Analyzing {', '.join(tqdm_fields_print)}")
            j += 1

    return results.reset_index(drop=True)




















# ============================================
# MAIN RUN
# ============================================


if __name__ == "__main__":
    
    print("="*80)
    print("Starting SCEA plume detection script...")

    # Configuration
    config_file = None
    config = load_config(config_file)

    # Extract variables from config (before logging setup for output directory)
    io_cfg = config.get("io_variables", {})
    variable_names_cfg = config.get("variable_names", {})
    preprocessing_steps_cfg = config.get("preprocessing_steps", [])
    scea_algo_cfg = config.get("scea_parameters_algorithmic", {})
    scea_runtime_cfg = config.get("scea_parameters_runtime", {})
    output_cfg = config.get("output_fields", [])
    verbose = config.get("verbose", True)
    tqdm_enabled = config.get("tqdm_enabled", True)
    step_by_step = config.get("step_by_step", False)
    step_plot_cfg = config.get("step_by_step_plotting", {})
    original_quality_threshold = _quality_filter_threshold(preprocessing_steps_cfg)

    # Generate hashes for config and input to track provenance and ensure reproducibility
    config_hash = get_config_hash(config, length=5)
    input_hash = get_input_hash(io_cfg, length=5)
    
    # Create temporary run_id for logging (will be updated after metadata is collected)
    temp_run_id = f"temp_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    
    # Setup logging
    console_level = parse_log_level(config.get("logger_console_level", "INFO"))
    logger = setup_logging(
        io_cfg["output_dir"], 
        temp_run_id, 
        level=console_level,
        console_output=config.get("logger_console_output", True)
    )
    logger.info("="*80)
    logger.info("SCEA Plume Detection Analysis Started")
    logger.info("="*80)
    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Input hash: {input_hash}")
    if step_by_step:
        logger.info("Step-by-step mode is enabled.")
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib/Cartopy is not installed. Interactive debug plots will be skipped.")

    # Validate config # TODO
    #validate_config(config)

    # import SCEA
    sys.path.append(os.path.dirname(io_cfg["scea_path"]))
    import SCEA_v3 as SCEA
    logger.info("Successfully imported SCEA_v3 module")

    # Get info about input files
    files_info = discover_input_files(io_cfg, logger=logger)

    # Get info about wind data
    # TODO this now takes a lot of time
    wind_data = None
    wind_data_info = None
    if "half_ellipse" in scea_algo_cfg["metric"]:
        wind_data, wind_data_info = load_wind_data(io_cfg, scea_algo_cfg, logger=logger, verbose=verbose)

    # Collect metadata for this run
    metadata = collect_run_metadata(config_file, config, config_hash, input_hash, files_info, wind_data_info, code_version=None, logger=logger, verbose=verbose)
    run_id = metadata["run_id"]
    
    # Update logger with final run_id
    logger.info(f"Final run ID: {run_id}")
    logger.info("="*80)

    # Run SCEA on files
    output_file = os.path.join(io_cfg["output_dir"], f"scea_results_{run_id}.csv")
    logger.info(f"Output CSV file: {output_file}")
    
    first_write = True

    tqdm_initial = tqdm(total=1, desc=" ", bar_format="{desc}", position=0, leave=True) if tqdm_enabled else None
    tqdm_initial.set_description("="*120)

    total_files = len(files_info["files_path"])
    tqdm_main = tqdm(total=total_files, desc="Files", unit="file", position=1, colour="blue") if tqdm_enabled else None

    # === MAIN LOOP ====
    for i in range(len(files_info["files_path"])):
        file_id = SCEA.file_id(filename=files_info["files_list"][i], mode="Sentinel-5P")


        # update per-file summary in the same line
        if tqdm_main:
            tqdm_main.set_description(f"FILE i={i}/{total_files-1} ({file_id})")
            tqdm_main.set_postfix({
                "file_id": file_id,
                #"time": "12.3s",
                #"clusters": 5
            })
            tqdm_main.update(1)


        if tqdm_enabled:
            preprocess_names = [step_cfg['name'][:12] for step_cfg in preprocessing_steps_cfg if step_cfg['enabled']]
            tqdm_preprocessing = tqdm(total=1, bar_format="{desc}", desc=f"  [0]{preprocess_names}", leave=False, position=2)
        else:
            tqdm_preprocessing = None

        tqdm_results = tqdm(total=1, bar_format="{desc}", leave=False, position=4) if tqdm_enabled else None

        tqdm_scea = tqdm(total=1, bar_format="{desc}", leave=False, position=3) if tqdm_enabled else None

        if tqdm_initial:
            tqdm_initial.set_description("="*120)
        if tqdm_preprocessing:
            preprocess_progress = ["⏳" + step_cfg['name'][:11] for step_cfg in preprocessing_steps_cfg if step_cfg['enabled']]
            tqdm_preprocessing.set_description(f"  [0]Preprocessing ({', '.join(preprocess_progress)})")
        if tqdm_scea:
            tqdm_scea.set_description(f"  [0]SCEA")
        if tqdm_results:
            tqdm_results.set_description(f"  [0]Analyzing {','.join(["⏳" + field_cfg['name'][:2] for field_cfg in output_cfg if field_cfg['enabled']])})")


        # Get data from file
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i+1}/{len(files_info['files_path'])}] Processing file {file_id}: {files_info['files_path'][i]}")
        logger.info(f"{'='*80}")

        try:
            #coords, values, time = get_data_from_file(file_path, variable_names=io_cfg["variable_names"], open_file_kwargs=io_cfg["open_file_kwargs"])
            logger.debug(f"Loading dataset from {files_info['files_path'][i]}")
            data_raw = xr.open_dataset(files_info["files_path"][i], **io_cfg["open_file_kwargs"])
            logger.debug(f"Data shape: {dict(data_raw.sizes)}")

            # If shape [1, x, y], normalize to [x, y] for SCEA processing. If leading dim >1, fail fast to avoid silently dropping data.
            data = normalize_singleton_leading_dim(data_raw, variable_names_cfg, logger=logger, verbose=verbose)
            original_data_for_debug = data.copy(deep=True)

            debug_dir = os.path.join(io_cfg["output_dir"], "step_by_step_debug", run_id, file_id)

            def _on_preprocess_step_complete(step_idx, step_cfg, before_data, after_data, original_data):
                if _should_skip_preprocessing_step_plot(step_cfg["name"], step_plot_cfg):
                    logger.debug(f"Skipping debug plot for preprocessing step '{step_cfg['name']}'.")
                    return

                plot_preprocessing_step_debug(
                    original_data=original_data,
                    before_data=before_data,
                    after_data=after_data,
                    variable_names_cfg=variable_names_cfg,
                    original_quality_threshold=original_quality_threshold,
                    step_name=step_cfg["name"],
                    step_kwargs=step_cfg.get("kwargs", {}),
                    debug_dir=debug_dir,
                    run_id=run_id,
                    file_id=file_id,
                    step_index=step_idx,
                    plotting_cfg=step_plot_cfg,
                    logger=logger,
                )


            # Preprocess file with provided preprocessing steps in config
            data = preprocess_for_scea(
                data,
                preprocessing_steps_cfg,
                variable_names_cfg,
                logger=logger,
                verbose=verbose,
                tqdm=tqdm_preprocessing,
                step_by_step=step_by_step,
                original_data=original_data_for_debug,
                on_step_complete=_on_preprocess_step_complete if step_by_step else None,
            )
            if tqdm_preprocessing:
                tqdm_preprocessing.refresh()

            # If data is None after preprocessing, skip to next file
            if data is None:
                logger.warning(f"File {file_id} skipped due to preprocessing failure.")
                continue

            
            wind_data_tropomi = None
            scaled_wind_magnitudes_linear = None
            wind_data_tropomi_angle = None
            if "half_ellipse" in scea_algo_cfg["metric"]:
                logger.debug("Matching wind data to TROPOMI coordinates")
                # TODO: add functionality to other data sources than Tropomi
                wind_data_tropomi = SCEA.match_era5_wind_to_tropomi(tropomi_data=data, era5_data=wind_data)
                wind_magnitude = "wind_magnitude_" + scea_algo_cfg["distance_matrix_kwargs"]["wind_height"]
                wind_angle = "wind_angle_" + scea_algo_cfg["distance_matrix_kwargs"]["wind_height"]
                logger.debug(f"Computing wind scaling using {wind_magnitude}")
                scaled_wind_magnitudes_linear = SCEA.scale_wind_magnitude_for_distance_matrix(wind_data_tropomi[wind_magnitude], method="linear", parameter=scea_algo_cfg["distance_matrix_kwargs"]["scaling_parameter"])
                wind_data_tropomi_angle = wind_data_tropomi[wind_angle].data.flatten()

            # Run SCEA
            logger.info("Running SCEA cluster detection...")
            clusters = SCEA.scea(
                coords = [data[variable_names_cfg["lon"]], data[variable_names_cfg["lat"]]],
                values = data[variable_names_cfg["value"]],
                growth_limit=scea_algo_cfg["growth_limit"],
                detection_limit=scea_algo_cfg["detection_limit"],
                max_pts_start_radius=scea_algo_cfg["max_pts_start_radius"],
                local_box_size=scea_algo_cfg["local_box_size"],
                metric=scea_algo_cfg["metric"],
                point_value_threshold=scea_algo_cfg["point_value_threshold"],
                distance_matrix_kwargs={
                    "rotation": wind_data_tropomi_angle,
                    "magnitude": scaled_wind_magnitudes_linear,
                },
                row_cache_max_rows=scea_runtime_cfg["row_cache_max_rows"],
                verbose=scea_runtime_cfg["verbose"],
                tqdm=tqdm_scea
            )
            n_clusters_detected = int(np.nanmax(clusters)) if np.nanmax(clusters) > 0 else 0
            logger.info(f"SCEA detected {n_clusters_detected} clusters")
            if tqdm_scea:
                tqdm_scea.refresh()

            if step_by_step:
                plot_clustering_debug(
                    original_data=original_data_for_debug,
                    preprocessed_data=data,
                    clusters=clusters,
                    variable_names_cfg=variable_names_cfg,
                    original_quality_threshold=original_quality_threshold,
                    debug_dir=debug_dir,
                    run_id=run_id,
                    file_id=file_id,
                    plotting_cfg=step_plot_cfg,
                    logger=logger,
                )


            logger.debug("Analyzing cluster outputs...")
            cluster_results = analyze_scea_clusters(
                clusters,
                output_config=output_cfg,
                lon=data[variable_names_cfg["lon"]].values,
                lat=data[variable_names_cfg["lat"]].values,
                values=data_raw[variable_names_cfg["value"]].values[0],
                timestamps=data[variable_names_cfg["time"]].values,
                qa_values=data[variable_names_cfg["quality"]].values,
                ignore_zero=True,
                logger=logger,
                verbose=verbose,
                filename=files_info['files_list'][i],
                tqdm=tqdm_results
            )
            if tqdm_results:
                tqdm_results.refresh()

            cluster_results["file_id"] = file_id
            cluster_results["file_name"] = os.path.basename(files_info["files_path"][i])

            if step_by_step:
                plot_cluster_outputs_debug(
                    preprocessed_data=data,
                    clusters=clusters,
                    cluster_results=cluster_results,
                    variable_names_cfg=variable_names_cfg,
                    debug_dir=debug_dir,
                    run_id=run_id,
                    file_id=file_id,
                    plotting_cfg=step_plot_cfg,
                    logger=logger,
                )

            # Save results to csv, appending if file already exists
            logger.debug(f"Writing {len(cluster_results)} cluster records to CSV")
            cluster_results.to_csv(
                output_file,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False
            logger.info(f"Successfully saved {len(cluster_results)} results to {output_file}")



            if tqdm_initial:
                tqdm_initial.refresh()
            if tqdm_preprocessing:
                tqdm_preprocessing.refresh()
            if tqdm_scea:
                tqdm_scea.refresh()
            if tqdm_results:
                tqdm_results.refresh()

        except Exception as e:
            logger.error(f"Error processing file {file_id}: {str(e)}", exc_info=True)
            tqdm.write(f"Error processing file {file_id}: {str(e)}. Check log for details. Skipping file.")
            if tqdm_scea:
                tqdm_scea.set_description(f"Error processing file {file_id}: {str(e)}")
            continue

    logger.info("="*80)
    logger.info(f"File processing complete. Results saved to {output_file}")
    logger.info("="*80)

    # Save metadata
    metadata_file = os.path.join(io_cfg["output_dir"], f"scea_metadata_{run_id}.yaml")
    try:
        with open(metadata_file, "w") as f:
            yaml.dump(metadata, f)
        logger.info(f"Saved run metadata to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}", exc_info=True)

    logger.info("="*80)
    logger.info("SCEA Plume Detection Analysis Completed Successfully")
    logger.info("="*80)

    print()
    if tqdm_initial:
        tqdm_initial.close()
    if tqdm_preprocessing:
        #tqdm_preprocessing.set_description(" ")
        #tqdm_preprocessing.refresh()
        tqdm_preprocessing.close()
    if tqdm_scea:
        #tqdm_scea.set_description(" ")
        tqdm_scea.close()
    if tqdm_results:
        #tqdm_results.set_description(" ")
        #tqdm_results.refresh()
        tqdm_results.close()
    if tqdm_main:
        tqdm_main.close()
    print()

    time.sleep(1) # ensure all logging output is flushed before printing final message

    print()
    print("Done!")
    print(f"Run ID: {run_id}")

