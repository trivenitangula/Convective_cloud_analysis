from importlib.resources import files
import logging
import matplotlib.pyplot as plt
import xarray as xr
import boto3
import io
import os
import numpy as np
import pandas as pd
import organization_indices  # Ensure objects.py is in the same folder
import imageio.v2 as iio
import glob
import matplotlib.dates as mdates
from botocore.exceptions import ClientError
import netCDF4
import seaborn as sns
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL


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


def main():
    """ Main function to process satellite data, compute organization indices, and visualize results. 
    Steps:
    1. Configuration: Set up paths, bucket names, and target date filters.
    2. Initialization: Create S3 client and read CSV labels.
    3. Data Processing: Loop through filtered dataset, read necessary files, compute indices for
       convective scenes, and store results.
    4. Visualization: Create KDE plots for each organization index, separated by cloud class."""

    bucket_name = "expats-msg-training"
    # S3 bucket containing the full domain WV files

    pat_dir = "/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN"
    # Base directory for the full domain files in S3

    basename = "merged_MSG_CMSAF"
    # Base name for the full domain files

    Ir108_bucket = "training-crops-ir108-100x100-140k"
    # S3 bucket containing the cropped IR108 files

    csv_file = "crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv"
    # CSV file with paths and labels for the cropped scenes

    s3 = init_s3()
    # Initialize S3 client

    df_labels = pd.read_csv(csv_file)
    # Read CSV labels into a DataFrame

    target_year = "2025"
    # Target year for filtering the dataset

    filtered_df = df_labels[(df_labels['path'].str.contains(
        f"/{target_year}-")) & (df_labels['path'].str.endswith('_1.nc'))].copy()
    # Filter the DataFrame for the target year and files ending with '_1.nc'

    balanced_df = filtered_df.groupby('label').head().copy()
    # Balance the dataset by taking an equal number of samples from each class label (up to 1000 per class)

    balanced_df = balanced_df.sort_values(by='path')
    # Sort by path to ensure chronological order

    print(
        f"Starting analysis for {len(balanced_df)} crops in {target_year})")
    # Log the number of crops being processed

    results = []
    # List to store results for each processed scene

    last_timestamp = None
    # Variable to track the last processed timestamp for efficient file reading

    ds_wv_full = None
    # Variable to hold the full domain WV dataset, reused for scenes with the same timestamp

    for index, row in balanced_df.head(35).iterrows():
        # Limit to first 2000 entries for testing

        file_path_from_csv = row['path']
        # Path to the cropped IR108 file in S3

        class_label = row['label']
        # Class label for the scene

        base_name = os.path.basename(file_path_from_csv)
        # Extract the base name of the file for timestamp parsing

        current_timestamp = base_name[:19]
        # Extract up to seconds for uniqueness

        date_part = base_name[:10]
        # Extract date part for S3 path construction

        yy, mm, dd = date_part.split('-')
        # Split date into components

        print(f"success! Extracted year:{yy}, month:{mm}, day:{dd}")
        # Log the extracted date components

        print(
            f"Processing File: {os.path.basename(file_path_from_csv)} | Class Label: {class_label}")
        # Log the file being processed and its class label

        if current_timestamp != last_timestamp:
            # Check if we need to read a new full domain WV file

            wv_file_key = f"{pat_dir}/{yy}/{mm}/{basename}_{yy}-{mm}-{dd}.nc"
            # Construct the S3 key for the full domain WV file based on the date

            wv_file_object = read_file(s3, wv_file_key, bucket_name)
            # Read the full domain WV file from S3

            if wv_file_object:
                # Check if the file was successfully read

                ds_wv_full = xr.open_dataset(io.BytesIO(wv_file_object))
                # Open the dataset from the file object

                last_timestamp = current_timestamp
                # Update the last timestamp to avoid redundant reads
            else:
                print(f"Skipping file: {file_path_from_csv}")
                continue
             # Skip if full domain WV file is missing

        ir108_object = read_file(s3, file_path_from_csv, Ir108_bucket)
        # Read the cropped IR108 file from S3

        if not ir108_object:
            print(f"Skipping file: {file_path_from_csv}")
            # Skip if cropped IR108 file is missing
            continue

        try:
            ds_ir = xr.open_dataset(io.BytesIO(ir108_object))
            # Open the cropped IR108 dataset from the file object

            ds_wv_cropped = ds_wv_full.reindex_like(
                ds_ir, method='nearest')
            # Crop the full domain WV dataset to match the IR108 grid using nearest neighbor interpolation

            ir108 = ds_ir.IR_108.values.squeeze()
            # Extract IR108 values as a 2D array

            wv062 = ds_wv_cropped.WV_062.values.squeeze()
            # Extract the cropped WV062 values as a 2D array
            BTD = ir108 - wv062
            # Calculate the Brightness Temperature Difference (BTD)

            if 'CMA' in ds_wv_cropped.variables:
                # Check if CMA variable exists in the cropped WV dataset
                cma = ds_wv_cropped.CMA.values.squeeze()
                # Extract CMA values if available
                cnv_idx = ((ir108 < 235) & (BTD > -5) & (cma == 1)).astype(int)
                # Define convective mask using IR, BTD, and CMA criteria
            else:
                cnv_idx = ((ir108 < 235) & (BTD > -5)).astype(int)
                # Define convective mask using only IR and BTD if CMA is not available

                nx = ds_ir.sizes['lon']
                # Get the number of longitude points from the IR dataset

                rmax = (nx / np.sqrt(2))
                # Calculate rmax based on the grid size

                dxy = 1
                # Define the bin width in pixels (assuming 1 pixel = 1 km)

                bins = np.arange(0, rmax+dxy, dxy)
                # Define bin edges for radial distribution

            # Calculate Centroids and number of convective objects using the defined mask
            centroids, ncnv = organization_indices._get_centroids(
                cnv_idx, periodic_BCs=False, periodic_zonal=False, clustering_algo=True
            )

            if ncnv >= 2:
                # Check if there are at least 2 convective objects to compute indices
                params = {
                    # Parameters for organization index computation
                    "dxy": dxy,
                    "bins": bins,
                    "cnv_idx": cnv_idx,
                    "rmax": rmax,
                    "periodic_BCs": False,
                    "periodic_zonal": False,
                    "clustering_algo": True,
                    "binomial_discrete": True,
                    "edge_mode": 'besag'
                }

                res = organization_indices._compute_organization_indices(
                    params)
                # Compute the organization indices using the defined parameters

                results.append({
                    # Store the results for the current scene
                    "path": file_path_from_csv,
                    "label": int(class_label),
                    "Lorg": res["L_org"],
                    "RIorg": res["RI_org"],
                    "OII": res.get("Oii", np.nan)
                })

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    results_df = pd.DataFrame(results)
    # Convert the results list into a DataFrame for analysis and visualization

    if results_df.empty:
        print("CRITICAL ERROR: No convective scenes (N >= 2) found in the selected period.")
        return
    # Exit if no valid scenes were processed

    results_df.to_csv(
        f"sample_Organization_Results_{target_year}.csv", index=False)
    # Save the results to a CSV file for record-keeping and further analysis

    print(f"Success! Processed {len(results_df)} scenes. Saved to CSV.")
    # Log the successful processing and saving of results

    class_colors = sns.color_palette("husl", 7)
    # Define a color palette for up to 7 classes (0-6)

    classes_sorted = sorted(results_df['label'].unique())
    # Get the unique class labels from the results for plotting

    metrics = ["OII", "RIorg", "Lorg"]
    # Define the organization indices to plot

    figure, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # Create a figure with 3 vertically stacked subplots for the three indices

    for ax, metric in zip(axes, metrics):
        # Loop through each subplot and corresponding metric

        for label in classes_sorted:
            # Loop through each class label

            subset = results_df[results_df['label'] == label]
            # Get the subset of results for the current class

            values = subset[metric].dropna().values
            # Extract the metric values, dropping any NaNs

            sample_count = len(values)
            # Get the number of samples for the current class and metric

            if sample_count < 2:
                continue
             # Skip plotting if there are fewer than 2 samples for KDE

            sns.kdeplot(
                values,
                ax=ax,
                label=f"Class {label} (n={sample_count})",

                color=class_colors[int(label)],
                linewidth=2.5
            )
            # Plot the KDE for the current class and metric with specified color and line width

        ax.set_title(f"{metric} Cloud Class Distribution",
                     fontsize=10, loc='right', fontweight='bold')
        # Set the title for each subplot with metric name and styling

        ax.set_xlabel(f"{metric}", fontsize=8, fontweight='bold')
        # Set the x-axis label for each subplot

        ax.set_ylabel("Density", fontsize=8, fontweight='bold')
        # Set the y-axis label for each subplot

        ax.grid(True, linestyle='--', alpha=0.6)
        # Add a grid to each subplot for better readability

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        # Place the legend outside the plot area to the right

    plt.tight_layout()
    # Adjust layout to prevent overlap of subplots and ensure everything fits well
    plt.savefig("Cloud_Class_Organization_samples.png",
                dpi=300, bbox_inches='tight')
    # Save the figure with high resolution and tight bounding box to ensure all elements are included
    plt.show()


if __name__ == "__main__":
    main()
