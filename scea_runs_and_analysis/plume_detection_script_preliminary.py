
import datetime
from glob import glob
import os
import yaml
import numpy as np
import pandas as pd
import sys
import copy


def preprocess_for_scea(file, preprocessing_steps_cfg):
    # TODO implement function to preprocess file and extract coordinates and values for SCEA input based on preprocessing_steps_cfg
    return None, None


def analyze_scea_clusters(clusters, output_config):
    # TODO implement function to analyze SCEA clusters and extract desired output fields based on output_config
    return




if __name__ == "__main__":

    # Configuration
    config_file = None

    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            print(f"Loaded config from {config_file}:")

            io_cfg = config.get("io_variables", {})
            preprocessing_steps_cfg = config.get("preprocessing_steps", [])
            scea_algo_cfg = config.get("scea_parameters_algorithmic", {})
            scea_runtime_cfg = config.get("scea_parameters_runtime", {})
            output_cfg = config.get("output_fields", [])
            verbose = config.get("verbose", True)

        if not isinstance(config, dict):
            raise ValueError("Loaded config must be a dictionary.")
        
    else:
        print("No config file provided, using default configuration in script.")

        # File variables
        io_cfg = {
            "files_dir": "",
            "files_type": "nc",
            "main_variable_name": "tropospheric_NO2_column_number_density",
            "wind_data_dir": "",
            "output_dir": "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA_developements/scea_runs_and_analysis/outputs",
            "scea_path": "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA_developements/SCEA_v3.py"
        }

        # Preprocessing
        preprocessing_steps_cfg = [
            {"name": "denoise_butterworth", "enabled": True, "kwargs": {"cutoff": 0.95, "order": 1}},
            {"name": "denoise_gaussian", "enabled": True, "kwargs": {"sigma": 1}},
            {"name": "local_standardization_m_mad", "enabled": True, "kwargs": {"window_size": 17}},
            {"name": "denoise_butterworth", "enabled": True, "kwargs": {"cutoff": 0.90, "order": 2}},
        ]

        # Detection: SCEA parameters
        scea_algo_cfg = {
            "growth_limit": 2,
            "detection_limit": 3.5,
            "max_pts_start_radius": 5,
            "local_box_size": 4,
            "metric": "geodesic_half_ellipse",
            "point_value_threshold": "stds_from_median",
            "distance_matrix_kwargs": None,
        }
        scea_runtime_cfg = {
            "row_cache_max_rows": 256,
            "verbose": False
        }

        # Results to save
        output_cfg = [
            "plume_id", # file_id + plume_number
            "file_id",
            "plume_number",
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
            "is_on_land",
            "median_of_larger_box",
            "is_connected_to_other_plume",
            "connected_plume_ids",
            "mean_q_value",
        ]

        # Runtime parameters
        verbose = True

        config = {
            "io_variables": io_cfg,
            "preprocessing_steps": preprocessing_steps_cfg,
            "scea_parameters_algorithmic": scea_algo_cfg,
            "scea_parameters_runtime": scea_runtime_cfg,
            "output_fields": output_cfg,
            "verbose": verbose,
        }





    # Input validation
    # TODO




    # Metadata
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc)
    code_version = None # TODO: add code to get version from git or similar

    print(f"Starting SCEA run with ID: {run_id}")
    print(f"Timestamp (UTC): {timestamp_utc.isoformat()}")
    print(f"Code version: {code_version}\n")

    # get info about files
    files_list = sorted(glob(os.path.join(io_cfg["files_dir"], f"*.{io_cfg['files_type']}")))
    files_time_range = None # TODO: implement time range extraction
    files_data_version = None # TODO: implement data version extraction

    print(f"Found {len(files_list)} files to process.")
    print(f"Example file: {files_list[0] if files_list else 'No files found'}")
    print(f"Time range of files: {files_time_range}")
    print(f"Data version of files: {files_data_version}\n")

    wind_files_list = None
    wind_time_range = None
    wind_data_version = None

    if "half_ellipse" in scea_algo_cfg["metric"]:
        # Get info about wind data
        wind_files_list = sorted(glob(os.path.join(io_cfg["wind_data_dir"], f"*.{io_cfg['files_type']}")))
        wind_time_range = None # TODO: implement time range extraction for wind data
        wind_data_version = None # TODO: implement data version extraction for wind data

        print(f"Found {len(wind_files_list)} wind data files.")
        print(f"Time range of wind data files: {wind_time_range}")
        print(f"Data version of wind data files: {wind_data_version}\n")


    # save metadata
    metadata = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc.isoformat(),
        "code_version": code_version,
        "config_file": config_file,
        "config": copy.deepcopy(config),
        "files_time_range": files_time_range,
        "files_data_version": files_data_version,
        "n_files": len(files_list),
        "files_list": files_list,
        "wind_data_source": None, # TODO: add info about wind data source (e.g. model name, resolution, etc.)
        "wind_time_range": wind_time_range,
        "wind_data_version": wind_data_version,
        "wind_files_list": wind_files_list,
    }


    # import SCEA
    sys.path.append(os.path.dirname(io_cfg["scea_path"]))
    from SCEA_v3 import SCEA

    results = []
    # Run SCEA on files
    for file in files_list:
        file_id = None # TODO: implement file_id extraction from filename or metadata

        # Preprocess file
        coords, values = preprocess_for_scea(file, preprocessing_steps_cfg) # TODO: implement function to preprocess file and extract coordinates and values for SCEA input

        # Run SCEA
        clusters = SCEA(
            coords = coords,
            values = values,
            growth_limit=scea_algo_cfg["growth_limit"],
            detection_limit=scea_algo_cfg["detection_limit"],
            max_pts_start_radius=scea_algo_cfg["max_pts_start_radius"],
            local_box_size=scea_algo_cfg["local_box_size"],
            metric=scea_algo_cfg["metric"],
            point_value_threshold=scea_algo_cfg["point_value_threshold"],
            distance_matrix_kwargs=scea_algo_cfg["distance_matrix_kwargs"],
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