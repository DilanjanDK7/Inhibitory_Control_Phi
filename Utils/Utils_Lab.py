import numpy as np
import pyphi
from scipy import signal
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pyphi.compute import phi





def tpm_le(time_series, clean=False, diag_off=True):
    """
    Constructs a transition probability matrix (TPM) from the time series.

    Args:
    - time_series (numpy array): A series of time points.
    - clean (bool): Remove rows with fewer than 5 transitions.
    - diag_off (bool): Set diagonal elements to 0.

    Returns:
    - TPM: The transition probability matrix.
    - state_total: Total transitions for each state.
    - frequency: Frequency of each state in the time series.
    """

    time_series = time_series.astype(np.int)
    markov_chain = time_series.tolist()
    n = len(markov_chain[0])

    # Initialize TPM with zeros
    tpm = np.zeros((2 ** n, 2 ** n))

    for (s1, s2) in zip(markov_chain, markov_chain[1:]):
        i, j = pyphi.convert.state2le_index(s1), pyphi.convert.state2le_index(s2)
        tpm[i][j] += 1

    # Cleaning: Set rows with insufficient transitions to zero
    if clean:
        tpm[np.sum(tpm, axis=1) < 5] = 0

    # Optionally turn off diagonal elements
    if diag_off:
        np.fill_diagonal(tpm, 0)

    # Normalize the TPM rows by the total number of transitions for each state
    state_total = np.sum(tpm, axis=-1)
    non_zero_rows = state_total != 0
    tpm[non_zero_rows] /= state_total[non_zero_rows][:, None]

    # Calculate the frequency of each state in the time series
    frequency = np.bincount([pyphi.convert.state2le_index(s) for s in markov_chain], minlength=2 ** n) / len(
        markov_chain)

    return np.copy(tpm), np.copy(state_total), np.copy(frequency)


def distance_arrays(E, O, method=0):
    """
    Computes the difference between two arrays using different methods.

    Args:
    - E (numpy array): Expected array.
    - O (numpy array): Observed array.
    - method (int): Distance calculation method.
        0 - squared difference,
        1 - absolute difference,
        2 - Frobenius norm,
        3 - Normalized Frobenius norm.

    Returns:
    - dif (float): The computed distance.
    """

    if method == 0:  # Squared difference, handling division by zero
        dif = np.nansum(np.where(E != 0, (E - O) ** 2 / E, 0))

    elif method == 1:  # Absolute difference
        dif = np.sum(np.abs(E - O))

    elif method == 2:  # Frobenius norm
        dif = np.linalg.norm(E - O, 'fro')

    elif method == 3:  # Normalized Frobenius norm
        dif = np.linalg.norm(E - O, 'fro') / np.linalg.norm(E, 'fro')

    return dif


def rms_single(ts):
    """
    Calculates the root mean square (RMS) of a time series.

    Args:
    - ts (numpy array): Input time series.

    Returns:
    - rms (float): The root mean square value.
    """
    return np.sqrt(np.mean(np.square(ts)))




def binary_ts(ts_single, cutoff_factor=None, median=True):
    """
    Converts a time series into a binary format based on statistical thresholds.

    Args:
    - ts_single (numpy array): Input time series.
    - cutoff_factor (float): Factor to scale the cutoff based on peak distribution.
    - median (bool): If True, use median as the threshold.

    Returns:
    - ts_binary (list): The binarized time series.
    """

    # Find peaks and valleys
    peaks = signal.find_peaks(ts_single)[0]
    valleys = signal.find_peaks(-ts_single)[0]
    peaks_all_ind = np.sort(np.concatenate((peaks, valleys)))

    # Calculate the amplitude threshold
    peak_amps = np.abs(ts_single[peaks_all_ind])
    if cutoff_factor:
        base_amp = np.median(peak_amps) + cutoff_factor * np.std(peak_amps)
    else:
        base_amp = np.median(peak_amps)

    # Initialize the binary time series
    ts_binary = np.zeros_like(ts_single)
    for i in peaks_all_ind:
        ts_binary[i:] = 1 if np.abs(ts_single[i]) > base_amp else 0

    return ts_binary


import numpy as np
from scipy import signal


def binary_ts_regions(ts_matrix, cutoff_factor=None, median=True):
    """
    Converts a 2D time series matrix (regions, time) into binary format based on statistical thresholds.

    Args:
    - ts_matrix (numpy array): Input 2D time series matrix with shape (regions, time).
    - cutoff_factor (float): Factor to scale the cutoff based on peak distribution.
    - median (bool): If True, use median as the threshold.

    Returns:
    - ts_binary_matrix (numpy array): The binarized time series with shape (regions, time).
    """

    # Initialize the binary time series matrix
    ts_binary_matrix = np.zeros_like(ts_matrix)

    # Iterate through each region's time series
    for region_idx in range(ts_matrix.shape[0]):
        ts_single = ts_matrix[region_idx]

        # Find peaks and valleys
        peaks = signal.find_peaks(ts_single)[0]
        valleys = signal.find_peaks(-ts_single)[0]
        peaks_all_ind = np.sort(np.concatenate((peaks, valleys)))

        # Calculate the amplitude threshold
        peak_amps = np.abs(ts_single[peaks_all_ind])
        if cutoff_factor:
            base_amp = np.median(peak_amps) + cutoff_factor * np.std(peak_amps)
        else:
            base_amp = np.median(peak_amps)

        # Binarize the time series for the current region
        ts_binary = np.zeros_like(ts_single)
        for i in peaks_all_ind:
            ts_binary[i:] = 1 if np.abs(ts_single[i]) > base_amp else 0

        # Store the binary time series in the matrix
        ts_binary_matrix[region_idx] = ts_binary

    return ts_binary_matrix


def binary_simple(ts, down_sample=None):
    """
    Converts a time series into a binary format based on comparison with mean + standard deviation.

    Args:
    - ts (numpy array): Time series data (rows: time points, columns: signals).
    - down_sample (int): Factor for down-sampling (optional).

    Returns:
    - ts_bin (numpy array): Binarized time series.
    """

    # Calculate mean and standard deviation for each signal
    mean_activity = np.mean(ts, axis=0)
    std_activity = np.std(ts, axis=0)
    
    # Define upper and lower thresholds
    upper_threshold = mean_activity + std_activity
    lower_threshold = mean_activity - std_activity
    
    # Binarize with two states only:
    # - Values above mean + std = 1 (high activity)
    # - Values below mean - std = 0 (low activity)
    # - Values between thresholds = 0 (neutral, grouped with low activity)
    ts_bin = (ts >= upper_threshold).astype(int)

    # Optional down-sampling
    if down_sample:
        ts_bin = ts_bin[::down_sample, :]

    return ts_bin


def get_binary_net(ts, band=[4, 8], down_sample=16, sf_in=256, out_length=6000, method="orig", n_ts=32):
    """
    Generates a binary network from a time series using a bandpass filter and various methods.

    Args:
    - ts (numpy array): Input time series.
    - band (list): Bandpass frequency range [low, high].
    - down_sample (int): Down-sampling factor.
    - sf_in (int): Sampling frequency.
    - out_length (int): Desired length of the output.
    - method (str): Method to compute the binary series ('orig', 'power', 'env').
    - n_ts (int): Number of time steps for the 'power' method.

    Returns:
    - ts_out (numpy array): Binarized network.
    """

    # Apply bandpass filter
    b, a = signal.butter(5, band, btype="bandpass", fs=sf_in)
    ts_filtered = signal.lfilter(b, a, ts, axis=0)

    # Choose binarization method
    method_map = {
        "orig": binary_ts,
        "power": lambda x: binary_power(x, n_ts=n_ts),
        "env": binary_env
    }

    ts_bin = np.vstack([method_map[method](ts_filtered[:, col]) for col in range(ts_filtered.shape[1])])

    # Down-sample for 'orig' method
    if method == "orig":
        if len(ts_bin) // down_sample >= out_length:
            ts_bin = ts_bin[::down_sample]
        else:
            raise ValueError("Time series is not long enough for the chosen sample rate and length.")

    return ts_bin.T


def binary_power(ts_single, n_ts, sample_freq=256):
    """
    Binarizes a time series based on the power (RMS) in time windows.

    Args:
    - ts_single (numpy array): Input time series.
    - n_ts (int): Number of time steps in each window.
    - sample_freq (int): Sampling frequency (default=256).

    Returns:
    - ts_bin (list): Binarized series based on windowed power.
    """

    n_win = len(ts_single) // n_ts
    baseline_all = rms_single(ts_single)  # Overall RMS

    ts_bin = []
    for t in range(n_win):
        baseline_single = rms_single(ts_single[t * n_ts: (t + 1) * n_ts])
        ts_bin.append(1 if baseline_single >= baseline_all else 0)

    return ts_bin


def binary_env(ts_single, sample_freq=256, check_plot=False):
    """
    Binarizes a time series based on the amplitude envelope.

    Args:
    - ts_single (numpy array): Input time series.
    - sample_freq (int): Sampling frequency (default=256).
    - check_plot (bool): If True, generates a plot comparing the original signal and the binary version.

    Returns:
    - ts_bin (numpy array): Binarized time series based on the envelope.
    """

    env = get_envelope(ts_single, fs=sample_freq)
    baseline = np.mean(env)

    ts_bin = (env >= baseline).astype(int)

    if check_plot:
        import matplotlib.pyplot as plt

        start, end = 0, 1000
        plt.subplot(211)
        plt.plot(ts_single[start:end], "b", label="Original")
        plt.plot(env[start:end], "r", label="Envelope")
        plt.hlines(baseline, start, end, "k", linestyles="dashed", label="Baseline")
        plt.legend()

        plt.subplot(212)
        plt.plot(ts_bin[start:end], "k", label="Binarized")
        plt.legend()
        plt.show()

    return ts_bin



def fft_signal(ts_single, sample_rate=256):
    """
    Computes and plots the FFT of a time series.

    Args:
    - ts_single (numpy array): Input time series.
    - sample_rate (int): Sampling rate (default=256).

    Returns:
    - None (plots the FFT).
    """

    N = len(ts_single)
    yf = fft(ts_single)
    xf = fftfreq(N, 1 / sample_rate)

    plt.plot(xf[:N // 2], np.abs(yf[:N // 2]))
    plt.title("FFT of the Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()


def get_envelope(ts_single, fs=256):
    """
    Computes the amplitude envelope of a time series using the Hilbert transform.

    Args:
    - ts_single (numpy array): Input time series.
    - fs (int): Sampling frequency (default=256).

    Returns:
    - amplitude_envelope (numpy array): The amplitude envelope of the input signal.
    """

    analytic_signal = hilbert(ts_single)
    amplitude_envelope = np.abs(analytic_signal)

    return amplitude_envelope


def plot_prob(tpm, cutoff=0.05):
    """
    Extracts probabilities from a transition probability matrix (TPM) above a given cutoff.

    Args:
    - tpm (numpy array): Transition probability matrix.
    - cutoff (float): Minimum threshold for extracted probabilities (default=0.05).

    Returns:
    - array_out (list): A list of probabilities that exceed the cutoff value.
    """

    array_out = tpm[tpm >= cutoff].tolist()

    return array_out


def swap_rows(timeseries):
    """
    Randomly swaps two rows of the time series.

    Args:
    - timeseries (numpy array): Input time series (rows as time points).

    Returns:
    - ts (numpy array): Time series with two randomly swapped rows.
    """

    ts = timeseries.copy() if timeseries.shape[0] > timeseries.shape[1] else timeseries.T.copy()

    rand_rows = np.random.choice(len(ts), size=2, replace=False)
    ts[rand_rows] = ts[rand_rows[::-1]]

    return ts


def condense_dir_list(dir_list):
    """
    Cleans up a list of directory entries by removing unwanted or duplicate entries.

    Args:
    - dir_list (list): List of directory entries.

    Returns:
    - list_new (list): Cleaned and condensed list without unwanted files.
    """

    list_new = []
    exclude_files = {".DS_Store", "._"}  # Set of unwanted files

    for d in dir_list:
        if not d.startswith("._") and d != ".DS_Store" and d not in list_new:
            list_new.append(d)

    return list_new




def to_calculate_mean_phi(tpm, spin_mean, eps=None, out_type="All"):
    """
    Computes the mean 'phi' value for a system based on its TPM and state probabilities.

    Args:
    - tpm (numpy array): Transition probability matrix.
    - spin_mean (numpy array): State probabilities.
    - eps (float): Threshold for ignoring states (optional).
    - out_type (str): 'All' to return all phi values, or 'Mean' to return the average.

    Returns:
    - phi_values (list or float): List of all phi values or their weighted mean.
    """

    network = pyphi.Network(tpm)
    n_states = len(tpm)

    state_indices = np.arange(n_states)
    states = np.array([pyphi.convert.le_index2state(idx, network.size) for idx in state_indices])

    phi_values = []
    for idx, state in enumerate(states):
        if eps is None or spin_mean[idx] > eps:
            subsystem = pyphi.Subsystem(network, state, range(network.size))
            phi_value = phi(subsystem)
            phi_values.append(phi_value)

    weighted_phi = np.sum(np.array(phi_values) * spin_mean[spin_mean > 0])

    if out_type == "All":
        phi_values.append(weighted_phi)
        return phi_values
    else:
        return np.mean(phi_values), weighted_phi
