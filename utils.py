from glob import glob

import numpy as np
import xarray as xr
import boto3
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
from credentials_buckets import S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY
import organization_indices
import matplotlib.pyplot as plt
import objects  # CRITICAL: This was missing!
import matplotlib.dates as mdates


def read_file(s3, file_path, bucket):
    """ Reads a file from the specified S3 bucket and returns its content.
    :param s3: An initialized S3 client.
    :param file_path: The path to the file within the S3 bucket.
    :param bucket: The name of the S3 bucket.
    :return: The content of the file if it exists, otherwise None.
    """

    try:
        obj = s3.get_object(Bucket=bucket, Key=file_path)
        return obj['Body'].read()
    except ClientError as e:
        logging.warning(f"Failed to read file {file_path}: {e}")
        return None


def init_s3():
    """
    Initializes and returns an S3 client using the provided credentials and endpoint.
    """

    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    )


def save_variable_to_netcdf(variable_data, time, units, var_name_31, output_path_31):
    """ Saves the given variable data to a NetCDF file at the specified output path.
    :param variable_data: The data to be saved, expected to be in a format compatible with xarray.
    :param time: The time coordinates for the data.
    :param units: The units of the variable.
    :param var: The name of the variable.
    :param output_path: The path where the NetCDF file will be saved.

    How to call the function:
    variable_data = [...]  # Your variable data here
    time = [...]  # Corresponding time coordinates
    units = "..."  # Units of the variable
    output_path = "path/to/output_file.nc"
    save_variable_to_netcdf(variable_data, time, units, output_path)

    """
    ds_out = xr.Dataset(
        {var_name_31: (['time'], variable_data)
         },
        coords={
            'time': time,
            'units': units
        }
    )
    ds_out.to_netcdf(output_path_31)
    print(f"Success! {var_name_31} saved to {output_path_31}")


def save_variable_to_netcdf(obs, theo, distance, output_path_1):
    """ Saves the given variable data to a NetCDF file at the specified output path.
    :param variable_data: The data to be saved, expected to be in a format compatible with xarray.
    :param distance: The distance coordinates for the data.
    :param units: The units of the variable.
    :param var: The name of the variable.
    :param output_path: The path where the NetCDF file will be saved.

    How to call the function:
    variable_data = [...]  # Your variable data here
    distance = [...]  # Corresponding distance coordinates
    units = "..."  # Units of the variable
    output_path = "path/to/output_file.nc"
    save_variable_to_netcdf(variable_data, distance, units, output_path)

    """
    ds_out = xr.Dataset(
        {"Besag_obs_1": (["distance"], obs),
         "Besag_theo_1": (["distance"], theo),
         },
        coords={
            'distance': distance,
        }
    )
    ds_out.to_netcdf(output_path_1)
    print(f"Success! Variables saved to {output_path_1}")


def read_and_process_data(ds):
    ir108 = ds.IR_108.values.astype(float)
    wv062 = ds.WV_062.values.astype(float)
    BTD = ir108 - wv062
    mask = ds.cma.values
    cnv_idx = ((ir108 < 230) & (BTD > -5) & (mask == 1)).astype(int)
    return cnv_idx


def read_and_compute_indices(ds, rmax, bins, dxy):
    cnv_idx = read_and_process_data(ds)

    # COMPUTE THE L-FUNCTION
    params = {
        "cnv_idx": cnv_idx,
        "rmax": rmax,
        "bins": bins,
        "periodic_BCs": False,
        "periodic_zonal": False,
        "clustering_algo": True,
        "binomial_continuous": False,
        "binomial_discrete": True,
        "edge_mode": 'besag',
        "dxy": dxy
    }
    res = organization_indices._compute_organization_indices(params)
    return res


def grid_parameters():
    dxy = 1
    rmax = 400
    bins = np.arange(0, rmax + dxy, dxy)
    return dxy, rmax, bins


def datasets(satellite_data_path):
    files = [
        r"C:\Users\trive\OneDrive\Desktop\hello world\my_satellite_data_20180611.nc",
        r"C:\Users\trive\OneDrive\Desktop\hello world\my_satellite_data_20210715.nc",
        r"C:\Users\trive\OneDrive\Desktop\hello world\my_satellite_data_20240621.nc",
        r"C:\Users\trive\OneDrive\Desktop\hello world\my_satellite_data_20120804.nc",
        r"C:\Users\trive\OneDrive\Desktop\hello world\my_satellite_data_20140802.nc",
        r"C:\Users\trive\OneDrive\Desktop\hello world\my_satellite_data_20240629.nc"
    ]
    return files


def target_timestamps():
    return ['06:00:00', '09:00:00', '12:00:00',
            '15:00:00', '18:00:00', '21:00:00']
