# Utils/Core_Phi.py

import numpy as np
import pyphi

# Function to discretize time series data
def discretize_time_series(time_series, n_bins):
    """
    Discretizes the time series into specified number of bins.

    Parameters:
    - time_series: A numpy array representing the time series data.
    - n_bins: Number of bins to discretize the data.

    Returns:
    - discretized_series: A numpy array with integer labels corresponding to bins.
    """
    discretized_series = np.digitize(time_series, np.histogram_bin_edges(time_series, bins=n_bins)) - 1
    return discretized_series


def tpm_le(time_series, clean=False, diag_off=True):
    """
    Generate a Transition Probability Matrix (TPM) from a given binary time series.

    Parameters:
    -----------
    time_series : numpy.ndarray
        A 2D array representing binary states over time. Each row is a time step, and each column corresponds to an element of the state.
    clean : bool, optional (default=False)
        If True, rows with fewer than 5 total transitions will be zeroed out.
    diag_off : bool, optional (default=True)
        If True, the diagonal of the TPM will be set to zero, preventing self-transitions.

    Returns:
    --------
    tpm : numpy.ndarray
        The normalized transition probability matrix of shape (2^n, 2^n), where n is the number of elements in each state.
    state_total : numpy.ndarray
        An array containing the total number of transitions from each state.
    frequency : numpy.ndarray
        The frequency of each state occurring in the time series.
    """

    # Ensure time_series is in integer format and initialize variables
    time_series = time_series.astype(np.int32)  # Use np.int32 for more explicit integer conversion
    n = time_series.shape[1]  # Number of elements in each state

    # Initialize the Transition Probability Matrix (TPM)
    tpm = np.zeros((2 ** n, 2 ** n))

    # Create the Markov chain by iterating through consecutive state pairs
    for s1, s2 in zip(time_series, time_series[1:]):
        i = pyphi.convert.state2le_index(s1)
        j = pyphi.convert.state2le_index(s2)
        tpm[i][j] += 1

    # Optionally clean rows where there are fewer than 5 transitions
    if clean:
        row_sums = np.sum(tpm, axis=1)
        tpm[row_sums < 5] = 0  # Set rows with fewer than 5 transitions to zero

    # Optionally remove self-transitions by setting diagonal to zero
    if diag_off:
        np.fill_diagonal(tpm, 0)

    # Calculate the total transitions from each state
    state_total = np.sum(tpm, axis=-1)

    # Compute the frequency of each state
    frequency = np.zeros(2 ** n)
    for s in time_series:
        i = pyphi.convert.state2le_index(s)
        frequency[i] += 1
    frequency /= len(time_series)

    # Normalize TPM by dividing each row by the total number of transitions from that state
    for r in range(len(state_total)):
        if state_total[r] != 0:
            tpm[r, :] /= state_total[r]  # Normalize only if there are transitions from the state

    return np.copy(tpm), np.copy(state_total), np.copy(frequency)


def construct_tpm(discretized_data):
    """
    Constructs a Transition Probability Matrix (TPM) from discretized data without assuming conditional independence.

    Parameters:
    - discretized_data: 2D numpy array of shape (T, n_regions), binary or multi-state.

    Returns:
    - tpm: 2^n_regions x 2^n_regions numpy array representing the joint TPM.
    """
    n_regions = discretized_data.shape[1]
    n_states = int(np.max(discretized_data) + 1)
    tpm = np.zeros((n_states ** n_regions, n_states ** n_regions))

    for t in range(len(discretized_data) - 1):
        current_state = tuple(discretized_data[t])
        next_state = tuple(discretized_data[t + 1])

        current_index = np.ravel_multi_index(current_state, [n_states] * n_regions)
        next_index = np.ravel_multi_index(next_state, [n_states] * n_regions)

        tpm[current_index, next_index] += 1

    # Normalize to get probabilities
    tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
    return tpm


def pyphi_analysis(tpm, n_states):
    """
    Performs basic PyPhi analysis on the TPM.

    Parameters:
    - tpm: Transition Probability Matrix.
    - n_states: Number of states per region.

    Returns:
    - phi: Integrated Information (Phi) value.
    - network: PyPhi Network object.
    """
    # Define the network using PyPhi
    network = pyphi.Network(tpm)

    # Define a current state (this example uses the first state, modify as needed)
    current_state = (0,) * network.size

    # Compute integrated information for the network
    subsystem = pyphi.Subsystem(network, current_state)
    phi = pyphi.compute.phi(subsystem)

    return phi, network


def compute_phi_at_scales(network, state):
    """
    Computes Phi for the whole system and all subsystems.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - phi_values: Dictionary containing Phi values for the whole system and all subsystems.
    """
    phi_values = {}

    # Compute Phi for the entire system
    whole_system = pyphi.Subsystem(network, state)
    phi_values['Whole System'] = pyphi.compute.phi(whole_system)

    # Compute Phi for each subsystem
    for subset in pyphi.utils.powerset(range(network.size)):
        if subset:
            subsystem = pyphi.Subsystem(network, state, subset)
            phi_values[f'Subsystem {subset}'] = pyphi.compute.phi(subsystem)

    return phi_values


def compute_mice(network, state):
    """
    Identifies Minimal Information Partitioned Elements (MICE) within the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - mice: List of tuples containing partitions and their corresponding MIP objects.
    """
    mice = []
    purview = list(range(network.size))  # Define purview as all node indices
    mechanism = purview  # Assuming mechanism is the entire purview

    # Iterate over all possible partitions for the given mechanism and purview
    try:
        partitions = pyphi.partition.all_partitions(mechanism)
    except TypeError as e:
        print(f"Error generating partitions: {e}")
        return mice

    for partition in partitions:
        subsystem = pyphi.Subsystem(network, state)
        try:
            mip = pyphi.compute.big_mip(subsystem, partition)
        except AttributeError:
            print("Error: 'big_mip' function not found in 'pyphi.compute'.")
            return mice
        except Exception as e:
            print(f"Error computing big_mip: {e}")
            continue

        if mip.phi > 0:
            mice.append((partition, mip))

    return mice


def identify_concepts_and_complexes(network, state):
    """
    Identifies Concepts and Complexes within the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - concepts: List of Concept objects.
    - complexes: List of Complex objects.
    """
    subsystem = pyphi.Subsystem(network, state)
    concepts = pyphi.compute.concepts(subsystem)
    complexes = pyphi.compute.complexes(network)

    return concepts, complexes


def cause_repertoire_analysis(network, state):
    """
    Analyzes the Cause Repertoire of the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - cause_repertoire: Cause repertoire of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    cause_repertoire = subsystem.cause_repertoire(pyphi.Direction.CAUSE)
    return cause_repertoire


def effect_repertoire_analysis(network, state):
    """
    Analyzes the Effect Repertoire of the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - effect_repertoire: Effect repertoire of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    effect_repertoire = subsystem.effect_repertoire(pyphi.Direction.EFFECT)
    return effect_repertoire


def cause_effect_space_exploration(network, state):
    """
    Explores the Cause-Effect Structure of the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - cause_effect_structure: Cause-effect structure of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    cause_effect_structure = pyphi.compute.cause_effect_structure(subsystem)
    return cause_effect_structure


def identify_major_complex(network, state):
    """
    Identifies the Major Complex within the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - major_complex: Major complex object.
    """
    subsystem = pyphi.Subsystem(network, state)
    major_complex = pyphi.compute.major_complex(subsystem)
    return major_complex


def partitioning_analysis(network, state):
    """
    Analyzes how different partitions of the network affect integrated information.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - phi_partitions: Dictionary mapping partitions to their corresponding Phi values.
    """
    subsystem = pyphi.Subsystem(network, state)
    purview = list(range(network.size))  # Define purview as all node indices
    mechanism = purview  # Assuming mechanism is the entire purview

    phi_partitions = {}
    # Iterate over all possible partitions
    for partition in pyphi.partition.all_partitions(mechanism):
        try:
            mip = pyphi.compute.big_mip(subsystem, partition)
        except AttributeError:
            print("Error: 'big_mip' function not found in 'pyphi.compute'.")
            continue
        except Exception as e:
            print(f"Error computing big_mip: {e}")
            continue
        phi_partitions[str(partition)] = mip.phi

    return phi_partitions


def state_transition_analysis(tpm):
    """
    Analyzes the State Transitions within the network's TPM.

    Parameters:
    - tpm: Transition Probability Matrix.

    Returns:
    - state_transitions: Dictionary mapping each state to its transition probabilities.
    """
    state_transitions = {}
    n_states = tpm.shape[0]

    for i in range(n_states):
        next_state_probs = tpm[i]
        state_transitions[f"State {i}"] = next_state_probs

    return state_transitions


def causal_structure_analysis(network, state):
    """
    Analyzes the Causal Structure of the network.

    Parameters:
    - network: PyPhi Network object.
    - state: Tuple representing the current state of the network.

    Returns:
    - causal_structure: Causal structure of the subsystem.
    """
    subsystem = pyphi.Subsystem(network, state)
    causal_structure = pyphi.compute.causal_structure(subsystem)
    return causal_structure
