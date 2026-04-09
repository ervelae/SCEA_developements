
import datetime
from glob import glob
import os
import yaml
import numpy as np
import pandas as pd
import sys
import copy
import xarray as xr


def default_config():
    return {        
        # File variables
        "io_variables": {
            "files_dir": "/Volumes/One Touch/Sentinel-5p NO2 Europe/S5P_NO2_June2024",
            "files_type": "nc",
            "file_indices_to_extract": None, # e.g. [0, 1, 2] to extract first three files, or None to extract all files
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
            {"name": "crop_to_bbox", "enabled": True, "kwargs": {"bbox": [-10.0, 35.0, 30.0, 70.0]}}, # [min_lon, min_lat, max_lon, max_lat]
            {"name": "quality_filter", "enabled": True, "kwargs": {"threshold": 0.75}}, # TODO: add quality filtering based on a quality variable in the data (e.g. qa_value for TROPOMI)
            {"name": "discard_small_files", "enabled": True, "kwargs": {"min_valid_points": 10}},
            {"name": "denoise_butterworth", "enabled": True, "kwargs": {"cutoff": 40, "order": 1}},
            {"name": "nan_gaussian_filter", "enabled": True, "kwargs": {"sigma": 1}},
            {"name": "local_standardization_m_mad", "enabled": True, "kwargs": {"window": 17}},
            {"name": "denoise_butterworth", "enabled": True, "kwargs": {"cutoff": 40, "order": 2}},
        ],

        # Detection: SCEA parameters
        "scea_parameters_algorithmic": {
            "growth_limit": 2,
            "detection_limit": 3.5,
            "max_pts_start_radius": 5,
            "local_box_size": 4,
            "metric": "geodesic_half_ellipse",
            "point_value_threshold": "stds_from_median",
            "distance_matrix_kwargs": {"wind_height": "100m", "wind_scaling_method": "linear", "scaling_parameter": 0.04} # only used for "half_ellipse" metric, can include parameters for distance matrix calculation such as wind height and scaling parameter for wind influence on distance
        },
        "scea_parameters_runtime": {
            "row_cache_max_rows": 256,
            "verbose": True
        },

        # Results to save
        "output_fields": [
            "plume_id", # file_id + plume_number
            "plume_number",
            "file_id",
            "file_name",
            "n_points_in_file",
            "timestamp_utc",
            "n_points",
            "max_point_loc",
            "max_point_value",
            "mean_point_value",
            "median_point_value",
            "bbox",
            "area",
            "wind_magnitude",
            "wind_direction",
            "is_weekend",
            "is_on_land",
            "median_of_larger_box",
            "is_connected_to_other_plume",
            "connected_plume_ids",
            "mean_q_value",
        ],
        # Runtime parameters
        "verbose": True
    }



def load_config(config_file):
    if config_file is None:
        print("\n[-] No config file provided, using default configuration in script.")
        return default_config()
    else:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            print(f"[Init] Loaded config from {config_file}:")

        if not isinstance(config, dict):
            raise ValueError("Loaded config must be a dictionary.")
        
        return config


def validate_config(config):
    # TODO implement function to validate config and raise errors if invalid

    # - if metric is "half_ellipse", check that wind data is provided and valid
    pass


"""
def get_data_from_file(file_path, variable_names, open_file_kwargs=None):

    data = xr.open_dataset(file_path, open_file_kwargs or {})

    #if variable_name not in data:
    #    raise ValueError(f"Variable '{variable_name}' not found in file {file}. Available variables: {list(data.data_vars.keys())}")
    
    lon = data[variable_names["lon"]].values
    lat = data[variable_names["lat"]].values
    time = data[variable_names["time"]].values

    coords = [lon, lat]

    values = variable_names["value"]

    return data
"""


def normalize_singleton_leading_dim(data, variable_names_cfg, verbose=False):
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


def preprocess_for_scea(data, preprocessing_steps_cfg, variable_names_cfg):
    """
    data: xarray dataset containing the data to preprocess
    """

    for step_cfg in preprocessing_steps_cfg:
        if step_cfg["enabled"]:

            step_kwargs = step_cfg.get("kwargs", {}) # get kwargs for this step, or empty dict if not provided

            if verbose: print(f"[pre] Preprocessing: {step_cfg['name']} with kwargs: {step_kwargs}")

            data = SCEA.preprocess_data_xr(data, step_cfg["name"], step_kwargs, variable_names_cfg)

            if data is None:
                if verbose: print(f"[-] Data discarded after preprocessing step: {step_cfg['name']}. Skipping file.")
                return None

    return data


def analyze_scea_clusters(clusters, output_config):
    if verbose:
        print(f"Analyzing {clusters.max()} SCEA clusters.")
    
    # TODO implement function to analyze SCEA clusters and extract desired output fields based on output_config
    return []














if __name__ == "__main__":

    # Configuration
    config_file = None
    config = load_config(config_file)

    # Extract variables from config
    io_cfg = config.get("io_variables", {})
    variable_names_cfg = config.get("variable_names", {})
    preprocessing_steps_cfg = config.get("preprocessing_steps", [])
    scea_algo_cfg = config.get("scea_parameters_algorithmic", {})
    scea_runtime_cfg = config.get("scea_parameters_runtime", {})
    output_cfg = config.get("output_fields", [])
    verbose = config.get("verbose", True)

    # Validate config
    validate_config(config)


    # import SCEA
    sys.path.append(os.path.dirname(io_cfg["scea_path"]))
    import SCEA_v3 as SCEA




    # === Put this metadata colllection into its own function? ===

    # Metadata
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc)
    code_version = None # TODO: add code to get version from git or similar

    print(f"\n[Init] Starting SCEA run with ID: {run_id}")
    if verbose: print(f"[Init] Timestamp (UTC): {timestamp_utc.isoformat()}")
    if verbose: print(f"[Init] Code version: {code_version}\n")

    # get info about files
    files_path = sorted(glob(os.path.join(io_cfg["files_dir"], f"*.{io_cfg['files_type']}")))
    if io_cfg["file_indices_to_extract"] is not None:
        files_path = [files_path[i] for i in io_cfg["file_indices_to_extract"]]
    files_time_range = None # TODO: implement time range extraction
    files_data_version = None # TODO: implement data version extraction

    if verbose:
        print(f"[Init] Found {len(files_path)} files to process.")
        print(f"[Init] Example file: {files_path[0] if files_path else 'No files found'}")
        print(f"[Init] Time range of files: {files_time_range}")
        print(f"[Init] Data version of files: {files_data_version}\n")

    wind_files_list = None
    wind_time_range = None
    wind_data_version = None


    # Wind stuff
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




    # save metadata
    metadata = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc.isoformat(),
        "code_version": code_version,
        "config_file": config_file,
        "config": copy.deepcopy(config),
        "files_time_range": files_time_range,
        "files_data_version": files_data_version,
        "n_files": len(files_path),
        "files_path": files_path,
        "wind_data_source": None, # TODO: add info about wind data source (e.g. model name, resolution, etc.)
        "wind_time_range": wind_time_range,
        "wind_data_version": wind_data_version,
        "wind_files_list": wind_files_list,
    }














    results = []
    # Run SCEA on files
    for i in range(len(files_path)):
        file_id = None # TODO: implement file_id extraction from filename or metadata

        # Get data from file
        if verbose: print(f"\n[{i}] Processing file {file_id}: {files_path[i]}")

        #coords, values, time = get_data_from_file(file_path, variable_names=io_cfg["variable_names"], open_file_kwargs=io_cfg["open_file_kwargs"])
        data = xr.open_dataset(files_path[i], **io_cfg["open_file_kwargs"])

        # If shape [1, x, y], normalize to [x, y] for SCEA processing. If leading dim >1, fail fast to avoid silently dropping data.
        data = normalize_singleton_leading_dim(data, variable_names_cfg, verbose=verbose)

        # Preprocess file
        data = preprocess_for_scea(data, preprocessing_steps_cfg, variable_names_cfg)

        if data is None:
            #if verbose: print(f"[-] Data discarded after preprocessing. Skipping file.")
            continue

        if "half_ellipse" in scea_algo_cfg["metric"]:
            # TODO: add functionality to other data sources than Tropomi
            wind_data_tropomi = SCEA.match_era5_wind_to_tropomi(tropomi_data=data, era5_data=wind_data)
            wind_magnitude = "wind_magnitude_" + scea_algo_cfg["distance_matrix_kwargs"]["wind_height"]
            wind_angle = "wind_angle_" + scea_algo_cfg["distance_matrix_kwargs"]["wind_height"]
            scaled_wind_magnitudes_linear = SCEA.scale_wind_magnitude_for_distance_matrix(wind_data_tropomi[wind_magnitude].data.flatten(), method="linear", parameter=0.05)



        # Run SCEA
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
                "rotation": wind_data_tropomi[wind_angle].data.flatten(),
                "magnitude": scaled_wind_magnitudes_linear,
            },
            row_cache_max_rows=scea_runtime_cfg["row_cache_max_rows"],
            verbose=scea_runtime_cfg["verbose"]
        )

        cluster_results = analyze_scea_clusters(clusters, output_config=output_cfg) # TODO: implement function to analyze clusters and extract desired output fields


        results.extend(cluster_results)



    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(io_cfg["output_dir"], f"scea_results_{run_id}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

    # Save metadata
    metadata_file = os.path.join(io_cfg["output_dir"], f"scea_metadata_{run_id}.yaml")
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")


    # TODO summary