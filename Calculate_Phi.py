# Testing_full_pipeline_only_Phi.py

import os
import time
import numpy as np
import xarray as xr
from scipy.io import loadmat
import scipy
import pyphi
import logging
from Utils.Core_Phi_only_phi import (
    discretize_time_series_binary,
    construct_tpm,
    compute_phi_at_scales,
    identify_concepts_and_complexes,
    compute_state_transitions,
    cause_repertoire_analysis,
    effect_repertoire_analysis,
    cause_effect_space_exploration,
    identify_major_complex,
    partitioning_analysis,
    causal_structure_analysis,
    to_calculate_mean_phi,
    tpm_le,
    test_tpm
)

# Suppress the PyPhi welcome message
os.environ['PYPHI_WELCOME_OFF'] = 'yes'

# Configure logging for the main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_and_process_data(dest_dir, subjects, network_types, output_dir):
    """
    Loads .mat files, processes the data, and saves them as NetCDF files.

    Parameters
    ----------
    dest_dir : str
        Base directory containing subject data.
    subjects : list
        List of subject identifiers.
    network_types : list
        List of network types.
    output_dir : str
        Base output directory to save processed data.

    Returns
    -------
    dict
        Dictionary with keys as (subject, network_type) and values as xarray DataArrays.
    """
    dataarrays = {}

    for subject in subjects:
        subject_dir = os.path.join(dest_dir, subject, 'Task1')

        ### When Changing the Task condition Make sure to choose the correct Output Directory !!!!!!

        for network_type in network_types:
            network_dir = os.path.join(subject_dir, network_type)

            if not os.path.isdir(network_dir):
                logger.warning(f"Directory does not exist: {network_dir}")
                continue

            mat_files = [file for file in os.listdir(network_dir) if file.endswith('.mat')]
            if not mat_files:
                logger.warning(f"No .mat files found in {network_dir}")
                continue

            for file in mat_files:
                file_path = os.path.join(network_dir, file)
                logger.info(f'Loading {file_path}')

                try:
                    mat_data = loadmat(file_path)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue

                if 'TF' not in mat_data:
                    logger.warning(f"'TF' key not found in {file_path}")
                    continue

                tf_data = mat_data['TF']  # Expected shape: (n_regions, T, F)

                if tf_data.ndim != 3:
                    logger.warning(f"Unexpected TF data dimensions in {file_path}: {tf_data.shape}")
                    continue

                n_regions = tf_data.shape[0]
                region_labels = [f'Region_{i + 1}' for i in range(n_regions)]
                T = tf_data.shape[1]
                time_labels = np.arange(1, T + 1)

                # Handle 'Freqs' which may contain strings like 'delta', 'theta', etc.
                if 'Freqs' in mat_data:
                    freqs = mat_data['Freqs'].flatten()
                    logger.info(f"'Freqs' raw data shape: {mat_data['Freqs'].shape}")
                    logger.info(f"'Freqs' raw data: {freqs}")

                    freq_labels = []
                    for freq in freqs:
                        # Check if freq is array-like and extract the first element
                        if isinstance(freq, (list, np.ndarray, tuple)) and len(freq) > 0:
                            freq = freq[0]

                        try:
                            freq_float = float(freq)
                            freq_labels.append(freq_float)
                        except (ValueError, TypeError):
                            freq_labels.append(str(freq))

                    # Filter only the desired frequency labels
                    desired_labels = ['delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']
                    # desired_labels = ['delta', 'theta', 'alpha', 'beta', 'gamma1']
                    filtered_freq_labels = [label for label in freq_labels if label in desired_labels]

                    logger.info(f"Processed frequency labels for {file_path}: {filtered_freq_labels}")
                else:
                    # Generate default frequency labels if 'Freqs' is not found
                    freq_labels = [f'Freq_{i + 1}' for i in range(tf_data.shape[2])]
                    logger.info(f"No 'Freqs' key found. Generated frequency labels: {freq_labels}")
                    filtered_freq_labels = freq_labels  # Assuming all frequencies are used

                # Check if the number of frequency labels matches the data
                if len(filtered_freq_labels) != tf_data.shape[2]:
                    logger.warning(
                        f"Mismatch in frequency labels and data for {file_path}: "
                        f"{len(filtered_freq_labels)} labels vs {tf_data.shape[2]} data points."
                    )
                    if len(filtered_freq_labels) > tf_data.shape[2]:
                        # Truncate freq_labels to match tf_data.shape[2]
                        filtered_freq_labels = filtered_freq_labels[:tf_data.shape[2]]
                        logger.info(f"Truncated frequency labels to {len(filtered_freq_labels)} to match data.")
                    else:
                        # Pad freq_labels with default labels to match tf_data.shape[2]
                        additional_labels = [f'Freq_{i + 1}' for i in
                                             range(tf_data.shape[2] - len(filtered_freq_labels))]
                        filtered_freq_labels.extend(additional_labels)
                        logger.info(f"Padded frequency labels to {len(filtered_freq_labels)} to match data.")

                tf_data_array = xr.DataArray(
                    tf_data,
                    dims=["Region", "Time", "Frequency"],
                    coords={"Region": region_labels, "Time": time_labels, "Frequency": filtered_freq_labels},
                    name="TF_Data"
                )
                # Remove the 'gamma2' frequency from the 'Frequency' dimension
                tf_data_array = tf_data_array.sel(Frequency=tf_data_array.Frequency != 'gamma2')

                # Verify DataArray Integrity
                logger.info(f"DataArray dimensions: {tf_data_array.dims}")
                logger.info(f"DataArray coordinates: {tf_data_array.coords}")

                output_data_dir = os.path.join(output_dir, 'Data', subject, network_type)
                os.makedirs(output_data_dir, exist_ok=True)
                output_file = os.path.join(output_data_dir, f'{subject}_{network_type}.nc')

                try:
                    tf_data_array.to_netcdf(output_file)
                    logger.info(f'Saved DataArray to {output_file}')
                except Exception as e:
                    logger.error(f"Error saving NetCDF file {output_file}: {e}")
                    continue

                dataarrays[(subject, network_type)] = tf_data_array

    return dataarrays


def perform_pyphi_analysis(dataarrays, output_dir, analysis_mode, overwrite=False, num_regions_mode1=None, num_regions_mode3=5, num_frequencies_mode3=5):
    """
    Performs PyPhi analysis on the loaded data and saves the results based on the selected analysis mode.

    Parameters
    ----------
    dataarrays : dict
        Dictionary with keys as (subject, network_type) and values as xarray DataArrays.
    output_dir : str
        Base output directory to save Phi results.
    analysis_mode : int
        Selected analysis mode (1, 2, or 3).
    overwrite : bool, optional
        Whether to overwrite existing phi.csv files. Defaults to False.
    num_regions_mode1 : int, optional
        The required number of regions for Analysis Mode 1. If specified, TPMs not matching
        this number of regions will be skipped.
    num_regions_mode3 : int, optional
        The required number of regions for Analysis Mode 3. Defaults to 5.
    num_frequencies_mode3 : int, optional
        The required number of frequencies for Analysis Mode 3. Defaults to 5.
    """
    # Determine total number of TPMs to process
    total_tpm = 0
    for (subject, network_type), tf_data_array in dataarrays.items():
        if analysis_mode in [1, 2]:
            if analysis_mode == 1 and num_regions_mode1 is not None:
                if len(tf_data_array.coords['Region']) == num_regions_mode1:
                    total_tpm += 1
            else:
                total_tpm += 1
        elif analysis_mode == 3:
            # Check if the number of regions or frequencies matches the required number
            num_regions = len(tf_data_array.coords['Region'])
            num_frequencies = len(tf_data_array.coords['Frequency'])
            if num_regions == num_regions_mode3 or num_frequencies == num_frequencies_mode3:
                total_tpm += num_frequencies
            else:
                logger.warning(
                    f"Skipping TPM creation for {subject}-{network_type} in Mode 3 because "
                    f"number of regions ({num_regions}) != {num_regions_mode3} and "
                    f"number of frequencies ({num_frequencies}) != {num_frequencies_mode3}."
                )
    logger.info(f"Total TPMs to process: {total_tpm}")

    processed_tpm = 0
    cumulative_time = 0.0

    for (subject, network_type), tf_data_array in dataarrays.items():
        logger.info(f"Processing PyPhi analysis for Subject: {subject}, Network Type: {network_type}")

        if analysis_mode == 1:
            # Mode 1: TPMs for regions (sum of all frequencies)
            logger.info("Selected Analysis Mode 1: TPMs for Regions (Sum of Frequencies)")

            # Check number of regions if num_regions_mode1 is set
            if num_regions_mode1 is not None:
                actual_num_regions = len(tf_data_array.coords['Region'])
                if actual_num_regions != num_regions_mode1:
                    logger.warning(
                        f"Number of regions ({actual_num_regions}) does not match the configured "
                        f"number of regions ({num_regions_mode1}). Skipping TPM for {subject}-{network_type}."
                    )
                    continue  # Skip to the next (subject, network_type)
                else:
                    logger.info(
                        f"Number of regions ({actual_num_regions}) matches the configured "
                        f"number of regions ({num_regions_mode1}). Proceeding with TPM construction."
                    )

            # Sum across frequencies
            region_sum = tf_data_array.sum(dim="Frequency").values  # Shape: (n_regions, T)
            # Transpose to shape (T, n_regions)
            region_sum = region_sum.T
            # Discretize
            discretized_data = discretize_time_series_binary(region_sum)

            # Define output directory for Mode 1
            mode_desc = "Regions_Sum_Frequencies"
            output_phi_dir = os.path.join(output_dir, 'Phi_Results', subject, network_type, mode_desc)
            phi_file_path = os.path.join(output_phi_dir, 'phi.csv')

            # Check if phi.csv exists and decide whether to skip
            if not overwrite and os.path.isfile(phi_file_path):
                logger.info(f"phi.csv already exists at {phi_file_path}. Skipping analysis for this TPM.")
                continue  # Skip to the next (subject, network_type)

        elif analysis_mode == 2:
            # Mode 2: TPMs for frequencies (sum of all regions)
            logger.info("Selected Analysis Mode 2: TPMs for Frequencies (Sum of Regions)")
            # Sum across regions
            frequency_sum = tf_data_array.sum(dim="Region").values  # Shape: (F, T)
            # Transpose to shape (T, F)
            frequency_sum = frequency_sum.T
            # Discretize
            discretized_data = discretize_time_series_binary(frequency_sum)

            # Define output directory for Mode 2
            mode_desc = "Frequencies_Sum_Regions"
            output_phi_dir = os.path.join(output_dir, 'Phi_Results', subject, network_type, mode_desc)
            phi_file_path = os.path.join(output_phi_dir, 'phi.csv')

            # Check if phi.csv exists and decide whether to skip
            if not overwrite and os.path.isfile(phi_file_path):
                logger.info(f"phi.csv already exists at {phi_file_path}. Skipping analysis for this TPM.")
                continue  # Skip to the next (subject, network_type)

        elif analysis_mode == 3:
            # Mode 3: TPMs per frequency (regions as nodes)
            logger.info("Selected Analysis Mode 3: TPMs per Frequency (Regions as Nodes)")

            # Check if the number of regions or frequencies matches the required number
            num_regions = len(tf_data_array.coords['Region'])
            num_frequencies = len(tf_data_array.coords['Frequency'])
            if num_regions != num_regions_mode3 and num_frequencies != num_frequencies_mode3:
                logger.warning(
                    f"Skipping TPM creation for {subject}-{network_type} in Mode 3 because "
                    f"number of regions ({num_regions}) != {num_regions_mode3} and "
                    f"number of frequencies ({num_frequencies}) != {num_frequencies_mode3}."
                )
                continue  # Skip to the next (subject, network_type)

            # Iterate over each frequency
            for freq_idx, freq_label in enumerate(tf_data_array.coords['Frequency'].values):
                logger.info(f"Processing Frequency: {freq_label}")

                # Define output directory for this frequency
                output_phi_dir = os.path.join(output_dir, 'Phi_Results', subject, network_type,
                                              f'Frequency_{freq_label}')
                phi_file_path = os.path.join(output_phi_dir, 'phi.csv')

                # Check if phi.csv exists and decide whether to skip
                if not overwrite and os.path.isfile(phi_file_path):
                    logger.info(f"phi.csv already exists at {phi_file_path}. Skipping analysis for Frequency: {freq_label}.")
                    continue  # Skip to the next frequency

                # Start timing for this TPM
                start_time = time.time()

                # Extract data for the current frequency
                freq_data = tf_data_array.isel(Frequency=freq_idx).values  # Shape: (n_regions, T)
                # Transpose to shape (T, n_regions)
                freq_data = freq_data.T
                # Discretize
                discretized_data = discretize_time_series_binary(freq_data)
                if num_regions !=num_regions_mode3:
                    logger.warning(
                        f"Skipping TPM creation for {subject}-{network_type} in Mode 3 because "
                        f"number of regions ({num_regions}) != {num_regions_mode3} and "
                        f"number of frequencies ({num_frequencies}) != {num_frequencies_mode3}."
                    )
                    continue
                # Construct TPM
                try:
                    tpm, f_1, f_2 = tpm_le(discretized_data, clean=False, diag_off=False)
                    # Convert TPM from state by state to state by node
                    tpm_tc = pyphi.convert.to_2dimensional(pyphi.convert.state_by_state2state_by_node(tpm))
                except ValueError as ve:
                    logger.error(f"TPM construction failed for {subject}-{network_type}-Frequency-{freq_label}: {ve}")
                    processed_tpm += 1
                    continue
                except Exception as e:
                    logger.error(
                        f"Unexpected error during TPM construction for {subject}-{network_type}-Frequency-{freq_label}: {e}"
                    )
                    processed_tpm += 1
                    continue

                # Compute Phi values
                try:
                    phi_save = to_calculate_mean_phi(tpm_tc, f_2, out_type='All')
                except Exception as e:
                    logger.error(f"Error computing Phi for {subject}-{network_type}-Frequency-{freq_label}: {e}")
                    processed_tpm += 1
                    continue

                # End timing for this TPM
                end_time = time.time()
                duration = end_time - start_time
                cumulative_time += duration
                processed_tpm += 1

                # Calculate average time per TPM
                average_time = cumulative_time / processed_tpm if processed_tpm > 0 else 0
                remaining_tpm = total_tpm - processed_tpm
                remaining_time = remaining_tpm * average_time

                # Log timing information
                logger.info(f"Time taken for current TPM: {duration:.2f} seconds")
                logger.info(f"Estimated remaining time: {remaining_time:.2f} seconds")

                # Log and save results
                logger.info(f"\nPhi Results for {subject} - {network_type} - Frequency: {freq_label}:")

                # Save Phi results
                os.makedirs(output_phi_dir, exist_ok=True)
                try:
                    np.savetxt(os.path.join(output_phi_dir, 'phi.csv'), phi_save, delimiter=',')
                    np.savetxt(os.path.join(output_phi_dir, 'tpm_sbs.csv'), tpm, delimiter=',')
                    np.savetxt(os.path.join(output_phi_dir, 'tpm_sbn.csv'), tpm_tc, delimiter=',')
                    logger.info(f"Saved Phi results to {output_phi_dir}")
                except Exception as e:
                    logger.error(f"Error saving Phi results to {output_phi_dir}: {e}")

            continue  # Skip the rest of the loop for mode 3

        else:
            logger.error(f"Invalid Analysis Mode selected: {analysis_mode}. Please choose 1, 2, or 3.")
            continue

        # Start timing for this TPM
        start_time = time.time()

        # For modes 1 and 2, proceed with TPM construction and Phi computation
        try:
            tpm, f_1, f_2 = tpm_le(discretized_data, clean=False, diag_off=False)
            # Convert TPM from state by state to state by node
            tpm_tc = pyphi.convert.to_2dimensional(pyphi.convert.state_by_state2state_by_node(tpm))
        except ValueError as ve:
            logger.error(f"TPM construction failed for {subject}-{network_type}: {ve}")
            processed_tpm += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error during TPM construction for {subject}-{network_type}: {e}")
            processed_tpm += 1
            continue

        # Compute Phi values
        try:
            phi_save = to_calculate_mean_phi(tpm_tc, f_2, out_type='All')
        except Exception as e:
            logger.error(f"Error computing Phi for {subject}-{network_type}: {e}")
            processed_tpm += 1
            continue

        # End timing for this TPM
        end_time = time.time()
        duration = end_time - start_time
        cumulative_time += duration
        processed_tpm += 1

        # Calculate average time per TPM
        average_time = cumulative_time / processed_tpm if processed_tpm > 0 else 0
        remaining_tpm = total_tpm - processed_tpm
        remaining_time = remaining_tpm * average_time

        # Log timing information
        logger.info(f"Time taken for current TPM: {duration:.2f} seconds")
        logger.info(f"Estimated remaining time: {remaining_time:.2f} seconds")

        # Log and save results
        logger.info(f"\nPhi Results for {subject} - {network_type} ({mode_desc}):")

        # Save Phi results
        os.makedirs(output_phi_dir, exist_ok=True)
        try:
            np.savetxt(os.path.join(output_phi_dir, 'phi.csv'), phi_save, delimiter=',')
            np.savetxt(os.path.join(output_phi_dir, 'tpm_sbs.csv'), tpm, delimiter=',')
            np.savetxt(os.path.join(output_phi_dir, 'tpm_sbn.csv'), tpm_tc, delimiter=',')
            logger.info(f"Saved Phi results to {output_phi_dir}")
        except Exception as e:
            logger.error(f"Error saving Phi results to {output_phi_dir}: {e}")


def main():
    """
    Main execution pipeline for loading data and performing PyPhi analysis.
    """
    # =======================
    # Configuration Section
    # =======================

    # Set OVERWRITE to True to overwrite existing phi.csv files
    # Set to False to skip analysis if phi.csv already exists
    OVERWRITE = False  # <-- Added configuration option for overwrite

    # Set the required number of regions for Analysis Mode 1
    NUM_REGIONS_MODE1 = 5  # <-- Set your desired number of regions here

    # Set the required number of regions and frequencies for Analysis Mode 3
    NUM_REGIONS_MODE3 = 5
    NUM_FREQUENCIES_MODE3 = 5

    test_tpm()
    ANALYSIS_MODE = 3 # Set to 1, 2, or 3 based on desired analysis
    """
    Mode 1 = Regions as nodes
    Mode 2 = Frequency as nodes
    Mode 3 =  Regions as nodes within each Frequency Band 
    """

    # Define directories - UPDATE THESE PATHS FOR YOUR SYSTEM
    dest_dir = '/path/to/your/data/directory'  # Update this path
    output_dir = '/path/to/your/output/directory'  # Update this path

    # Define subjects and network types
    subjects = []  # Extend as needed
    network_types = [
        'frontoparietal',
        'Salience',
        'Visual',
        'Auditory',
        'cingulooperc',
        'Default',
        'retrosplinialtemporal',
        'smmouth',
        'smhand',
        'cinguloparietal',
        'Dorsalattn'
    ]

    # Validate ANALYSIS_MODE
    if ANALYSIS_MODE not in [1, 2, 3]:
        logger.error(f"Invalid ANALYSIS_MODE: {ANALYSIS_MODE}. Please set it to 1, 2, or 3.")
        return

    # Load and process data
    dataarrays = load_and_process_data(dest_dir, subjects, network_types, output_dir)

    # Perform PyPhi analysis with the overwrite option and number of regions/frequencies for Mode 1 and 3
    perform_pyphi_analysis(
        dataarrays,
        output_dir,
        ANALYSIS_MODE,
        overwrite=OVERWRITE,
        num_regions_mode1=NUM_REGIONS_MODE1 if ANALYSIS_MODE == 1 else None,
        num_regions_mode3=NUM_REGIONS_MODE3 if ANALYSIS_MODE == 3 else None,
        num_frequencies_mode3=NUM_FREQUENCIES_MODE3 if ANALYSIS_MODE == 3 else None
    )


if __name__ == "__main__":
    main()


""" 
"""