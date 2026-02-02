import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "significant_height_of_combined_wind_waves_and_swell",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "10m_u_component_of_neutral_wind",
        "10m_v_component_of_neutral_wind",
        "10m_wind_gust_since_previous_post_processing",
        "instantaneous_10m_wind_gust"
    ],
    "year": ["2024"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "07:00", "08:00", "09:00",
        "10:00", "11:00", "12:00",
        "13:00", "14:00", "15:00",
        "16:00", "17:00"
    ],
    "data_format": "grib",
    "download_format": "zip",
    "area": [75, -15, 30, 35]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
