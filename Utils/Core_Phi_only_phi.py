# Utils/Core_Phi.py

import numpy as np
import pyphi
import itertools
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def state_to_index(state):
    """
    Converts a binary state tuple to a linear index.

    Parameters
    ----------
    state : tuple
        A tuple representing the binary state, e.g., (0, 1, 0).

    Returns
    -------
    int
        The corresponding linear index.
    """
    return int(''.join(str(bit) for bit in state), 2)


def index_to_state(index, n):
    """
    Converts a linear index to a binary state tuple.

    Parameters
    ----------
    index : int
        The linear index.
    n : int
        Number of bits in the state.

    Returns
    -------
    tuple
        The corresponding binary state tuple.
    """
    return tuple(int(x) for x in bin(index)[2:].zfill(n))


def powerset(iterable):
    """
    Generates all non-empty subsets of the input iterable.

    Parameters
    ----------
    iterable : iterable
        The input iterable.

    Yields
    ------
    tuple
        Each non-empty subset as a tuple.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

def to_calculate_mean_phi(tpm, spin_mean, eps=None, out_type="All"):
    from pyphi.compute import phi
    """
    Calculates the mean Phi values for a given Transition Probability Matrix (TPM) and spin mean.

    Parameters
    ----------
    tpm : np.ndarray
        Transition Probability Matrix.
    spin_mean : np.ndarray
        Array of spin mean values corresponding to each state.
    eps : float, optional
        Threshold for including Phi values. States with spin_mean below this value are included. Defaults to None.
    out_type : str, optional
        Output type. If "All", returns all Phi values plus the sum. Otherwise, returns the mean and sum. Defaults to "All".

    Returns
    -------
    list or tuple
        Depending on `out_type`, returns either a list of Phi values including the sum or a tuple of mean Phi and the sum.
    """
    rows, columns = tpm.shape

    # Calculate the number of nodes based on the TPM size
    n_nodes = int(np.log2(rows))
    if 2**n_nodes != rows:
        raise ValueError(f"TPM size {rows} is not a power of 2. Cannot determine the number of nodes.")

    setting_int = np.linspace(0, rows - 1, num=rows).astype(int)

    # Dynamically use the calculated number of nodes instead of a hardcoded value
    M = list(map(lambda x: list(pyphi.convert.le_index2state(x, n_nodes)), setting_int))
    M = np.asarray(M).astype(int)

    phi_values = []

    network = pyphi.Network(tpm)
    for state in range(rows):
        if eps is None:
            if spin_mean[state] != 0:
                print("Current state:")
                print(M[state, :])
                phi_values.append(phi(pyphi.Subsystem(network, M[state, :], range(network.size))))
        else:
            if spin_mean[state] < eps:
                phi_values.append(phi(pyphi.Subsystem(network, M[state, :], range(network.size))))

    weight = spin_mean[spin_mean != 0]

    phiSum = np.sum(phi_values * weight)

    if out_type == "All":
        phi_values.append(phiSum)
        return phi_values
    else:
        return np.mean(phi_values), phiSum
def discretize_time_series(time_series, n_bins):
    """
    Discretizes the time series into a specified number of bins.

    Parameters
    ----------
    time_series : np.ndarray
        A numpy array representing the time series data.
    n_bins : int
        Number of bins to discretize the data.

    Returns
    -------
    np.ndarray
        Discretized time series with integer labels.
    """
    discretized = np.digitize(time_series, np.histogram_bin_edges(time_series, bins=n_bins)) - 1
    return discretized

def tpm_le(time_series, clean = False, diag_off = True):

    import numpy as np
    import pyphi

    time_series = time_series.astype(np.int)

    markov_chain = time_series.tolist()
    n = len(markov_chain[0])
    tpm = np.zeros((2 ** n, 2 ** n))

    for (s1, s2) in zip(markov_chain, markov_chain[1:]):
        i = pyphi.convert.state2le_index(s1)
        j = pyphi.convert.state2le_index(s2)
        tpm[i][j] += 1

    if clean == True:
        for r in range(2**n):
            if np.sum(tpm[r, :]) < 5:
                tpm[r, :] = np.zeros(2**n)

    if diag_off == True:
        for r in range(2**n):
            tpm[r, r] = 0

    state_total = np.sum(tpm, axis=-1)

    frequency = np.zeros((2 ** time_series.shape[-1]))

    for s in markov_chain:
        i = pyphi.convert.state2le_index(s)
        frequency[i] += 1

    frequency /= len(markov_chain)

    for div in range(len(state_total)):
        if state_total[div] != 0.0:
            tpm[div, :] /= state_total[div]

    return np.copy(tpm), np.copy(state_total), np.copy(frequency)

def discretize_time_series_binary(time_series):
    """
    Discretizes the time series into binary states based on the mean.

    Parameters
    ----------
    time_series : np.ndarray
        A 2D numpy array of shape (T, n_regions) or (n_regions, T).

    Returns
    -------
    np.ndarray
        Binary discretized time series of shape (T, n_regions).
    """
    # Determine the time axis (assume the larger dimension is time)
    time_axis = 0 if time_series.shape[0] > time_series.shape[1] else 1
    if time_axis != 0:
        time_series = time_series.T

    # Handle NaNs with interpolation
    if np.isnan(time_series).any():
        logger.warning("NaN values detected. Applying interpolation.")
        for region in range(time_series.shape[1]):
            nans = np.isnan(time_series[:, region])
            if np.any(nans):
                not_nans = ~nans
                if not np.any(not_nans):
                    time_series[:, region] = 0
                else:
                    time_series[nans, region] = np.interp(
                        np.flatnonzero(nans),
                        np.flatnonzero(not_nans),
                        time_series[not_nans, region]
                    )

    mean_activity = np.mean(time_series, axis=0, keepdims=True)
    discretized = (time_series > mean_activity).astype(int)
    return discretized


def construct_tpm(discretized_data):
    """
    Constructs a Transition Probability Matrix (TPM) from binary discretized data.

    Parameters
    ----------
    discretized_data : np.ndarray
        Binary discretized time series of shape (T, n_nodes).

    Returns
    -------
    np.ndarray
        The joint TPM of shape (2^n, 2^n), where n is the number of nodes.
    """
    n_nodes = discretized_data.shape[1]
    n_states = 2 ** n_nodes
    tpm = np.zeros((n_states, n_states), dtype=np.float64)

    # Convert binary states to integer indices
    powers = 1 << np.arange(n_nodes - 1, -1, -1)
    current_states = (discretized_data[:-1] * powers).sum(axis=1)
    next_states = (discretized_data[1:] * powers).sum(axis=1)

    # Count transitions
    for i, j in zip(current_states, next_states):
        tpm[i, j] += 1

    # Normalize TPM
    with np.errstate(divide='ignore', invalid='ignore'):
        tpm /= tpm.sum(axis=1, keepdims=True)
        tpm[np.isnan(tpm)] = 0  # Handle division by zero

    # Check for rows with all zeros
    rows_all_zero = np.where(tpm.sum(axis=1) == 0)[0]
    if len(rows_all_zero) > 0:
        logger.warning(f"{len(rows_all_zero)} rows in TPM have all zero transitions. Setting to uniform distribution.")
        tpm[rows_all_zero] = 1.0 / tpm.shape[1]  # Set to uniform distribution

    # Final TPM Integrity Check
    if not np.allclose(tpm.sum(axis=1), 1.0):
        logger.error("TPM normalization failed. Some rows do not sum to 1.")
        raise ValueError("TPM normalization failed. Some rows do not sum to 1.")

    logger.info("TPM constructed and normalized successfully.")

    # Final TPM Integrity Check
    if not np.allclose(tpm.sum(axis=1), 1.0):
        logger.error("TPM normalization failed. Some rows do not sum to 1.")
        raise ValueError("TPM normalization failed. Some rows do not sum to 1.")

    logger.info("TPM constructed and normalized successfully.")
    logger.info(f"TPM shape: {tpm.shape}")
    logger.info(f"Sample TPM rows:\n{tpm[:5]}")  # Log first 5 rows for inspection
    return tpm


def compute_phi(network, state):
    """
    Computes the integrated information (Phi) for a given network and state.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    float
        The Phi value.
    """
    subsystem = pyphi.Subsystem(network, state)
    phi = pyphi.compute.phi(subsystem)
    return phi


def compute_phi_at_scales(network, state):
    phi_values = {}

    # Whole system
    # phi_values['Whole System'] = compute_phi(network, state)
    # logger.info("Whole System")
    # logger.info(f"Phi for Whole System: {phi_values['Whole System']}")

    # Subsystems
    for subset in powerset(range(network.size)):
        if len(subset) == network.size:
            continue  # Skip the whole system as it's already computed
        subnetwork = network.subnetwork(subset)
        subsystem_state = tuple(state[i] for i in subset)
        subsystem = pyphi.Subsystem(subnetwork, subsystem_state)
        phi = pyphi.compute.phi(subsystem)
        phi_values[f'Subsystem {subset}'] = phi
        logger.info(f"Phi for Subsystem {subset}: {phi}")

    logger.info("Phi computed for all scales.")
    return phi_values


def identify_concepts_and_complexes(network, state):
    """
    Identifies Concepts and Complexes within the network.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    tuple
        A tuple containing lists of Concepts and Complexes.
    """
    subsystem = pyphi.Subsystem(network, state)
    concepts = pyphi.compute.concepts(subsystem)
    complexes = pyphi.compute.complexes(network)
    return concepts, complexes


def compute_state_transitions(tpm):
    """
    Analyzes the State Transitions within the network's TPM.

    Parameters
    ----------
    tpm : np.ndarray
        Transition Probability Matrix.

    Returns
    -------
    dict
        Dictionary mapping each state to its transition probabilities.
    """
    state_transitions = {}
    n_states = tpm.shape[0]

    for i in range(n_states):
        next_state_probs = tpm[i]
        state_transitions[f"State {i}"] = next_state_probs.tolist()

    return state_transitions


def cause_repertoire_analysis(network, state):
    """
    Analyzes the Cause Repertoire of the network.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    dict
        Cause repertoire of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    cause_repertoire = subsystem.cause_repertoire(pyphi.Direction.CAUSE)
    return cause_repertoire


def effect_repertoire_analysis(network, state):
    """
    Analyzes the Effect Repertoire of the network.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    dict
        Effect repertoire of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    effect_repertoire = subsystem.effect_repertoire(pyphi.Direction.EFFECT)
    return effect_repertoire


def cause_effect_space_exploration(network, state):
    """
    Explores the Cause-Effect Structure of the network.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    dict
        Cause-effect structure of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    cause_effect_structure = pyphi.compute.cause_effect_structure(subsystem)
    return cause_effect_structure


def identify_major_complex(network, state):
    """
    Identifies the Major Complex within the network.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    pyphi.Complex
        The major complex object.
    """
    subsystem = pyphi.Subsystem(network, state)
    major_complex = pyphi.compute.major_complex(subsystem)
    return major_complex


def partitioning_analysis(network, state):
    """
    Analyzes how different partitions of the network affect integrated information.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    dict
        Dictionary mapping partitions to their corresponding Phi values.
    """
    subsystem = pyphi.Subsystem(network, state)
    purview = list(range(network.size))  # Define purview as all node indices
    mechanism = purview  # Assuming mechanism is the entire purview

    phi_partitions = {}
    # Iterate over all possible partitions
    for partition in powerset(mechanism):
        try:
            mip = pyphi.compute.big_mip(subsystem, partition)
            phi_partitions[str(partition)] = mip.phi
        except Exception as e:
            logger.error(f"Error computing big_mip for partition {partition}: {e}")
            continue

    return phi_partitions


def causal_structure_analysis(network, state):
    """
    Analyzes the Causal Structure of the network.

    Parameters
    ----------
    network : pyphi.Network
        The PyPhi network object.
    state : tuple
        The current state of the network.

    Returns
    -------
    dict
        Causal structure of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    causal_structure = pyphi.compute.causal_structure(subsystem)
    return causal_structure



def test_tpm():
    # Simple TPM with conditional dependencies
    tpm = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    try:
        network = pyphi.Network(tpm)
        logger.info("Test TPM Network created successfully.")
    except Exception as e:
        logger.error(f"Error creating Test PyPhi Network: {e}")

