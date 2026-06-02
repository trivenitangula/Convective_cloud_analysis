import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import organization_indices
import matplotlib.colors as mcolors
from utils import datasets, read_file, read_and_process_data, read_and_compute_indices, grid_parameters, target_timestamps, datasets


def main():
    """ Main function to read satellite data, compute organization indices, and create comparison plots.
     This function reads a NetCDF dataset, processes it to compute organization indices for specific timestamps, 
     and generates a series of plots comparing observed and theoretical distributions, as well as a convective mask. 
     The resulting figure is saved as a PNG file.
    """
    dxy, rmax, bins = grid_parameters()
    # Get grid parameters

    dataset_name = "my_satellite_data_20210715.nc"
    ds = xr.open_dataset(dataset_name)

    cnv_idx = read_and_process_data(ds)
    # Read and process the convective index data from the dataset

    file_date = str(ds.time.values[0])[:10]
    # Extract the date from the dataset's time coordinate for timestamp construction

    timestamps = target_timestamps()
    # Get target timestamps for analysis

    fig, axes = plt.subplots(
        len(timestamps), 5, figsize=(20, 5*len(timestamps)))
    # Create a figure with subplots for each time step and each type of plot

    # Processes each time step in the dataset
    for i, t_suffix in enumerate(timestamps):

        timestamp = f"{file_date}T{t_suffix}"
        # Construct the full timestamp string for selection

        print(f"Processing timestamp: {timestamp}")
        # Print the current time step being processed
        try:
            timescale = ds.sel(
                time=timestamp, method='nearest', tolerance='15min')
            # Select the time step closest to the target timestamp with a tolerance of 15 minutes

            res = read_and_compute_indices(
                timescale, rmax, bins, dxy)
            # Compute the organization indices for the selected time step

            # Plotting the normalized nearest neighbor cumulative distribution function (NNCDF) comparison
            axes[i, 0].plot(res["NNCDF_theor"],
                            label='NNCDF Theoretical', color='orange')
            axes[i, 0].plot(res["NNCDF_obs"],
                            label='NNCDF Observed', color='red')
            axes[i, 0].set_title("NNCDF Comparison")
            axes[i, 0].set_xlabel("Distance (pixels)")
            axes[i, 0].set_xlim(0, rmax)  # Limit x-axis to focus on
            axes[i, 0].set_ylabel("Cumulative Distribution")
            axes[i, 0].grid(True)

            # Normalized distance for Besag plot
            axes[i, 1].plot(res["Besag_theor"], label='Besag Theoretical',
                            color='blue')
            axes[i, 1].plot(res["Besag_obs"], label='Besag Observed',
                            color='brown')
            axes[i, 1].set_title("Besag Comparison")
            axes[i, 1].set_xlabel("Distance (pixels)")
            axes[i, 1].set_xlim(0, rmax)  # Limit x-axis to focus on
            axes[i, 1].set_ylabel("$\\bar{L}(z)$")
            axes[i, 1].grid(True)

            #  convert distance bins to normalized distance z for the Besag plot
            z = bins/rmax

            # Plotting Besag comparison with normalized distance z
            axes[i, 2].plot(z, res["Besag_theor"], label='Besag Theoretical',
                            color='blue', alpha=0.7, linewidth=2)
            axes[i, 2].plot(z, res["Besag_obs"], label='Besag Observed',
                            color='brown', alpha=0.5, linewidth=2)

            # Reference line for random distribution
            axes[i, 2].plot(z, z, linestyle='--', linewidth=1,
                            label='$\\bar{L}(z)=z$', color='black', alpha=0.5)

            # Shading areas where observed is above or below theoretical
            axes[i, 2].fill_between(z, res["Besag_obs"], res["Besag_theor"], where=(
                res["Besag_obs"] > res["Besag_theor"]), interpolate=True, alpha=0.25, color='brown')
            axes[i, 2].fill_between(z, res["Besag_obs"], res["Besag_theor"], where=(
                res["Besag_obs"] < res["Besag_theor"]), interpolate=True, alpha=0.25, color='purple')

            # Convert normalized distance z back to pixels for secondary x-axis
            def z2pix(x):
                return x * rmax

            def pix2z(x):
                return x / rmax

            secax = axes[i, 2].secondary_xaxis('top', functions=(z2pix, pix2z))
            # Add a secondary x-axis on top of the Besag plot

            secax.set_xlabel('$r$ [pixels]')
            # Assuming rmax is in pixels
            secax.set_xticks(np.linspace(0, rmax, 3))
            # Set ticks at 0, rmax/2, and rmax
            axes[i, 2].set_title("Besag Comparison (Normalized)")
            axes[i, 2].set_ylim(0, 1)
            axes[i, 2].set_xlim(0, 1)
            axes[i, 2].set_xticks(np.arange(0, 1.1, 0.5))
            axes[i, 2].set_yticks(np.arange(0, 1.1, 0.5))
            axes[i, 2].set_xlabel("$z = r/r_{max}$")
            axes[i, 2].set_ylabel("$\\bar{L}(z)$")

            axes[i, 2].annotate(f"OII={res['Oii']:.4f}", xy=(0.80, 0.15), xycoords='axes fraction',
                                fontsize=10, ha='right', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            axes[i, 2].annotate(f"Lorg={res['L_org']:.4f}", xy=(0.80, 0.10), xycoords='axes fraction',
                                fontsize=10, ha='right', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3', fc='cyan', alpha=0.5))
            axes[i, 2].grid(True)
            axes[i, 2].legend(loc='upper left', fontsize='small')

            # Plotting the NNCDF comparison with confidence intervals and OII_2 annotation
            axes[i, 3].plot(res["NNCDF_theor"], label='NNCDF Theoretical',
                            color='blue', alpha=0.7)
            # Plot theoretical NNCDF with blue color and some transparency

            axes[i, 3].plot(res["NNCDF_obs"], label='NNCDF Observed',
                            color='brown', alpha=0.5)
            # Plot observed NNCDF with brown color and more transparency

            axes[i, 3].plot(res["NNCDF_theor"], res['NNCDF_theor'], linestyle='--', linewidth=1,
                            label='$\\bar{F}=F$', color='black', alpha=0.5)
            # Reference line for perfect agreement

            axes[i, 3].fill_between(res["NNCDF_theor"], res["NNCDF_obs"], res["NNCDF_theor"], where=(
                res["NNCDF_obs"] > res["NNCDF_theor"]), interpolate=True, alpha=0.25, color='brown')
            # Fill area where observed is above theoretical

            axes[i, 3].fill_between(res["NNCDF_theor"], res["NNCDF_obs"], res["NNCDF_theor"], where=(
                res["NNCDF_obs"] < res["NNCDF_theor"]), interpolate=True, alpha=0.25, color='blue')
            # Fill area where observed is below theoretical

            confidence_interval = 0.95
            degree_of_freedom = len(res["NNCDF_theor"]) - 1
            t_value = np.abs(np.random.standard_t(
                degree_of_freedom, size=1000))
            # for 95% confidence
            margin_of_error = np.percentile(t_value, 97.5) * np.std(
                res["NNCDF_obs"] - res["NNCDF_theor"]) / np.sqrt(len(res["NNCDF_theor"]))
            # confidence_interval = margin_of_error

            lower_bound = res["NNCDF_theor"] - margin_of_error
            # Calculate lower bound of confidence interval
            upper_bound = res["NNCDF_theor"] + margin_of_error
            # Calculate upper bound of confidence interval
            axes[i, 3].fill_between(res["NNCDF_theor"], lower_bound,
                                    upper_bound, color='gray', alpha=0.2, label='±95% Interval')
            # Fill confidence interval area
            axes[i, 3].legend(loc='upper left', fontsize='small')
            axes[i, 3].annotate(f"OII_2={res['Oii_2']:.4f}", xy=(0.80, 0.15), xycoords='axes fraction',
                                fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            # Annotate OII_2 value on the plot
            axes[i, 3].annotate(f"Iorg={res['RI_org']:.4f}", xy=(0.80, 0.10), xycoords='axes fraction',
                                fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='cyan', alpha=0.5))
            # Annotate RI_org value on the plot

            axes[i, 3].set_title("NNCDF Comparison (Normalized)")
            axes[i, 3].set_xlabel("$F$")
            axes[i, 3].set_ylabel("$\\bar{F}$")
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].set_xlim(0, 1)
            axes[i, 3].set_xticks(np.arange(0, 1.1, 0.5))
            axes[i, 3].set_yticks(np.arange(0, 1.1, 0.5))
            axes[i, 3].grid(True)

            cnv_idx_mask = mcolors.ListedColormap(['blue', 'red'])
            # Define a colormap for the convective index mask (0=blue, 1=red)

            norm_mask_cnv_idx = mcolors.BoundaryNorm(
                [0, 0.5, 1], cnv_idx_mask.N)
            # Define boundaries for the convective index mask

            lon = ds.lon.values
            lat = ds.lat.values
            current_cnv_idx = read_and_process_data(timescale)
            axes[i, 4].pcolormesh(
                lon, lat, current_cnv_idx, cmap=cnv_idx_mask, norm=norm_mask_cnv_idx)
            # Plot the convective index as a colored mesh

            colorbar = plt.colorbar(axes[i, 4].pcolormesh(
                lon, lat, current_cnv_idx, cmap=cnv_idx_mask, norm=norm_mask_cnv_idx), ax=axes[i, 4], ticks=[0.25, 0.75])
            # Add a colorbar for the convective index plot

            colorbar.set_label("Convective Index")
            axes[i, 4].set_title("Convective Mask")
            axes[i, 4].set_xlabel("Longitude")
            axes[i, 4].set_ylabel("Latitude")

        # Catch any exceptions during processing and print an error message
        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")

    fig.tight_layout()
    fig.savefig(f"Comparison_test_{dataset_name}.png", dpi=300)


if __name__ == "__main__":
    main()
