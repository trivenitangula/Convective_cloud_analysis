import os
from matplotlib import colors
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import organization_indices


def save_variable_to_netcdf(variable_data, time, labels, units, var_name_45, output_path_45):
    """ Saves the given variable data to a NetCDF file at the specified output path.
    :param variable_data: The dataset to be saved.
    :param time: The time coordinates for the data.
    :param labels: The class labels corresponding to each time point.
    :param units: The units of the variable.
    :param var_name_45: The name of the variable to be saved in the NetCDF file.
    :param output_path_45: The path where the NetCDF file will be saved.
    """
    ds_out = xr.Dataset(
        {
            var_name_45: (['time'], variable_data),
        },
        # Define the variable and its dimensions for the NetCDF dataset
        coords={
            'time': time,
            'units': units,
            'labels': (['time'], labels)
        }
        # Define the coordinates for time, units, and labels in the NetCDF dataset
    )
    ds_out.to_netcdf(output_path_45)
    # Save the dataset to a NetCDF file at the specified output path
    print(f"Variable data saved to {output_path_45}")
    # Print a confirmation message after saving the file
    return None


Original_file = "my_satellite_data_20210715.nc"
# Path to the original NetCDF file containing the satellite data for processing

Csv_file = (
    "crop_list_dcv2_resnet_k7_ir108_100x100_"
    "2013-2017-2021-2025_2xrandomcrops_"
    "1xtimestamp_cma_nc_all_140207.csv"
)
# Path to the CSV file containing the class labels and timestamps for the satellite data


def main():
    """ Main function to process the satellite data, compute organization indices, and plot the results.
    This function reads the original satellite data and corresponding labels, computes organization indices for each timestamp, saves the results to a NetCDF file, and creates a comparison plot of the Lorg index for different classes and timestamps.
    Steps:
    1. Read the original satellite data from a NetCDF file and the corresponding labels from a CSV file. 
    2. Extract timestamps from the CSV file and associate them with class labels.
    3. Compute organization indices (OII, Iorg, Lorg, RIorg) for each timestamp in the satellite data using the specified parameters.
    4. Save the computed RIorg index to a NetCDF file for later use in plotting.
    5. Create a comparison plot of the Lorg index for different classes and timestamps, using different markers and colors to differentiate between classes and timestamps.
    """
    ds_orig = xr.open_dataset(Original_file)
    # Open the original NetCDF dataset containing the satellite data for processing

    df_labels = pd.read_csv(Csv_file)
    # Read the CSV file containing the class labels and timestamps for the satellite data into a DataFrame

    path_column = "path"
    # The column in the CSV file that contains the file paths from which timestamps will be extracted

    df_labels[path_column] = df_labels[path_column].astype(str)
    # Ensure that the path column is treated as strings for consistent processing when extracting timestamps from file paths

    csv_timestamps = []
    # Initialize an empty list to store the extracted timestamps from the CSV file for each entry in the path column

    for p in df_labels[path_column]:
        # Loop through each file path in the specified column of the labels DataFrame to extract timestamps
        try:
            filename = os.path.basename(p)
            # Extract the filename from the full file path to facilitate timestamp extraction

            filename = os.path.splitext(filename)[0]
            # Remove the file extension from the filename to isolate the part containing the timestamp for easier parsing

            parts = filename.split("_")
            # Split the filename into parts using underscores as delimiters, which is expected to separate different components of the filename, including the timestamp

            timestamp_found = None
            # Initialize a variable to store the found timestamp, if any, during the parsing of the filename parts

            for part in parts:
                if "T" in part and "-" in part:
                    timestamp_found = pd.to_datetime(part)
                    break
                # Check if the part contains both "T" and "-", which are indicative of a timestamp format, and if found, convert it to a datetime object and break the loop

            csv_timestamps.append(timestamp_found)
            # Append the found timestamp (or None if not found) to the list of timestamps corresponding to each entry in the labels DataFrame for later association with class labels and plotting
        except Exception:
            csv_timestamps.append(pd.NaT)
            # If any error occurs during the timestamp extraction process, append a NaT (Not a Time) value to maintain the alignment of timestamps with the labels DataFrame, ensuring that the DataFrame structure remains intact even if some timestamps cannot be extracted successfully

    df_labels["timestamp"] = csv_timestamps
    # Add the extracted timestamps as a new column in the labels DataFrame, allowing for easy association of timestamps with class labels and facilitating subsequent analysis and plotting based on these timestamps

    print(df_labels[["timestamp", "label"]].head())
    # Print the first few rows of the labels DataFrame to verify that the timestamps have been extracted and associated with the correct class labels, providing a quick check on the success of the timestamp extraction process before proceeding with further analysis and plotting

    nx = ds_orig.sizes['lon']
    # Get the size of the longitude dimension from the original dataset to define the grid parameters for computing organization indices

    rmax = nx / np.sqrt(2)
    # Calculate the maximum radius (rmax) for the organization indices based on the size of the longitude dimension, using the formula rmax = nx / sqrt(2) to ensure that the maximum distance considered for the indices is appropriate for the grid size of the dataset

    dxy = 1
    # Define the grid spacing (dxy) for computing the organization indices, which determines the resolution at which distances are calculated for the indices

    bins = np.arange(0, rmax + dxy, dxy)
    # Create an array of distance bins from 0 to rmax with a spacing of dxy, which will be used for computing the cumulative distribution functions and other distance-based metrics in the organization indices calculations

    results = []
    # Initialize an empty list to store the computed organization indices and associated information for each timestamp, which will later be converted into a DataFrame for analysis and plotting

    print(f"\nProcessing {len(ds_orig.time)} timesteps")
    # Print the total number of timesteps in the original dataset to provide an overview of the processing workload before starting the loop to compute organization indices for each timestamp

    for i in range(len(ds_orig.time)):
        try:
            timescale = ds_orig.isel(time=i)
            # Select the data for the current timestamp by indexing the original dataset along the time dimension, allowing for the computation of organization indices for each individual time step in the dataset

            timestamp_dt = pd.to_datetime(timescale.time.values)
            # Convert the time value for the current timestamp to a pandas datetime object for easier handling and association with class labels in the subsequent analysis and plotting steps

            timestamp_str = timestamp_dt.strftime('%Y-%m-%dT%H:%M')
            # Format the timestamp as a string in the format "YYYY-MM-DDTHH:MM" for consistent display and logging during the processing of each timestamp, providing a clear and standardized representation of the time being processed in the output messages

            print(f"\n{timestamp_str}")

            ir108 = timescale.IR_108.values.astype(float)
            # Extract the IR_108 variable from the current timestamp's data and convert it to a float array for processing in the organization indices calculations

            wv062 = timescale.WV_062.values.astype(float)
            # Extract the WV_062 variable from the current timestamp's data and convert it to a float array for processing in the organization indices calculations

            cma = timescale.cma.values.astype(float)
            # Extract the CMA variable from the current timestamp's data and convert it to a float array for processing in the organization indices calculations

            btd = wv062 - ir108
            # Calculate the brightness temperature difference (BTD) between the WV_062 and IR_108 channels, which is often used in cloud detection and classification algorithms to identify convective objects based on their thermal properties

            cnv_idx = ((ir108 < 235) & (btd > -5) & (cma == 1)).astype(int)
            #  Create a binary index (cnv_idx) for convective objects based on the specified thresholds for IR_108, BTD, and CMA, where pixels that meet the criteria are marked as 1 (convective) and others as 0 (non-convective), which will be used as input for computing the organization indices

            centroids, ncnv = (
                organization_indices._get_centroids(cnv_idx, periodic_BCs=False,
                                                    periodic_zonal=False, clustering_algo=True
                                                    )
            )
            # Get the centroids of the convective objects identified in the cnv_idx using the specified parameters for boundary conditions and clustering algorithm, and also retrieve the number of convective objects (ncnv) which will be used to determine if there are enough objects to compute the organization indices that require at least 2 objects for meaningful calculations

            if ncnv < 2:
                print("Skipping few objects")
                # If there are fewer than 2 convective objects, print a message and skip the computation of organization indices for this timestamp, as certain indices (like Lorg) require at least 2 objects to be calculated meaningfully, ensuring that the results are not skewed by insufficient data points for those indices
                continue

            params = {
                "dxy": dxy,
                "cnv_idx": cnv_idx,
                "rmax": rmax,
                "bins": bins,
                "periodic_BCs": False,
                "periodic_zonal": False,
                "clustering_algo": True,
                "binomial_discrete": True,
                "edge_mode": "besag"
            }
            # Define the parameters for computing the organization indices, including grid spacing, convective index, maximum radius, distance bins, boundary conditions, clustering algorithm, and edge mode, which will be passed to the function that computes the organization indices for the current timestamp

            res = (organization_indices._compute_organization_indices(params))
            # Compute the organization indices for the current timestamp using the specified parameters

            oii = res["Oii"]
            # Extract the OII index from the results for the current timestamp

            iorg = res["I_org"]
            # Extract the Iorg index from the results for the current timestamp

            lorg = res["L_org"]
            # Extract the Lorg index from the results for the current timestamp

            riorg = res["RI_org"]
            # Extract the RIorg index from the results for the current timestamp

            print(f"OII={oii:.3f} | "f"Iorg={iorg:.3f} | "f"Lorg={lorg:.3f}")
            # Print the computed indices for the current timestamp

            matches = df_labels[df_labels["timestamp"] == timestamp_dt]
            # Find matching timestamp in the labels DataFrame to get the corresponding class label

            if not matches.empty:
                class_label = matches["label"].mode()[0]
            else:
                class_label = "Unknown"
                # Assign "Unknown" if no matching timestamp is found in the labels DataFrame

            results.append({
                "timestamp": timestamp_dt,
                "label": class_label,
                "OII": oii,
                "Iorg": iorg,
                "Lorg": lorg,
                "RIorg": riorg
            })
            # Append the computed indices and associated label for the current timestamp to the results list

        except Exception as e:
            print(f"ERROR: {e}")

    results_df = pd.DataFrame(results)
    # Convert the list of results into a DataFrame for easier analysis and plotting

    print(results_df.head())
    # Print the first few rows of the results DataFrame to verify the computed indices and associated labels

    results_df.to_csv("organization_metrics.csv", index=False)
    # Save the results to a CSV file for future reference and analysis

    save_variable_to_netcdf(
        variable_data=results_df["RIorg"].values,
        time=results_df["timestamp"].values,
        labels=results_df["label"].values,
        units="dimensionless",
        var_name_45="RIorg_rmax_20210715_temp235_195",
        output_path_45=r"C:\Users\trive\OneDrive\Desktop\hello world\RIorg_results_20210715_temp235_rmax195.nc"
    )
    # Save the RIorg variable to a NetCDF file for later use in plotting

    class_colors = {
        "Class 0": "#1f77b4",  # Blue
        "Class 1": "#ff7f0e",  # Orange
        "Class 2": "#2ca02c",  # Green
        "Class 3": "#d62728",  # Red
        "Class 4": "#9467bd",  # Purple
        "Class 5": "#8c564b",  # Brown
        "Class 6": "#7f7f7f"   # Gray
    }

    files = [
        r"C:\Users\trive\OneDrive\Desktop\hello world\Lorg_results_20210715_temp235_rmax195.nc",
        # Path to the first dataset
        r"C:\Users\trive\OneDrive\Desktop\hello world\Lorg_results_20250921_temp235_rmax195.nc",
        # Path to the second dataset
    ]
    file_labels = ["2021-07-15", "2025-09-21"]
    # Labels for the two datasets to be used in the legend

    df_labels_clean = (df_labels.dropna(subset=['timestamp']).groupby('timestamp')['label'].agg(lambda x: x.mode()[0])
                       .reset_index()
                       )
    # Clean the labels DataFrame to have one label per timestamp, using the mode for duplicates

    plt.figure(figsize=(12, 6))
    # Set the figure size for better visibility

    for file, f_label in zip(files, file_labels):
        # Loop through each file and its corresponding label for plotting
        try:
            ds = xr.open_dataset(file)
            # Open the NetCDF dataset for the current file
            var_names = list(ds.data_vars)
            # Get the list of variable names in the dataset

            if not var_names:
                print(f"Skipping {file}: No variables found.")
                continue
            # Skip if no variables are found in the dataset

            target_var = var_names[0]
            # Assuming the variable of interest is the first one in the dataset; adjust if necessary

            df_plot = pd.DataFrame({"time": pd.to_datetime(ds.time.values), "value": ds[target_var].values
                                    })
            # Create a DataFrame for plotting with time and the variable values from the dataset

            df_plot = pd.merge(df_plot, df_labels_clean,
                               left_on='time', right_on='timestamp', how='left')
            # Merge the plotting DataFrame with the cleaned labels DataFrame to associate class labels with timestamps

            df_plot["class"] = df_plot["label"].fillna("Unknown")
            # Fill any missing labels with "Unknown" to ensure all points are categorized for plotting

            df_plot = df_plot.sort_values("time")
            # Sort the DataFrame by time to ensure the plot is in chronological order

            df_plot["plot_x"] = df_plot["time"].apply(
                lambda t: t.replace(year=2000, month=1, day=1))
            # Normalize the time to a common date for better comparison on the x-axis, keeping only the time of day

            for class_name, group in df_plot.groupby("class"):
                # Loop through each class group for plotting

                class_str = str(class_name)
                # Convert class name to string for consistent handling

                if class_str.isdigit() or (class_str.replace('.', '', 1).isdigit()):
                    class_str = f"Class {int(float(class_str))}"
                    # Format numeric class labels as "Class X" for better readability in the legend

                color = class_colors.get(class_str, "#7f7f7f")  # Gray fallback

                marker_style = 'o' if f_label == "2021-07-15" else '^'
                # Use circles for the first dataset and triangles for the second dataset to differentiate them visually

                if "Class 2" in class_str:
                    marker_size = 100
                    # Make Class 2 significantly larger
                    marker_alpha = 0.9
                    # Make it less transparent so it pops out
                else:
                    marker_size = 50
                    # Default size for other classes
                    marker_alpha = 0.5
                    # Default transparency

                plt.scatter(group["plot_x"], group["value"], label=f"{class_str} ({f_label})", color=color, s=marker_size, marker=marker_style, alpha=marker_alpha
                            )

        except Exception as e:
            print(f"Error processing {file}: {e}")
            # Print any errors encountered while processing each file

    plt.title(
        "Lorg Comparison Chart", fontsize=14)
    plt.xlabel("Time of Day (UTC)", fontsize=12)
    plt.ylabel("$L_{org}$ Index", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()

    handles, plot_labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        print("Warning: The plot is empty.")
        # Print a warning if no data points were plotted

    plt.tight_layout()
    plt.savefig("Lorg_Class_Comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
