import numpy as np
import pandas as pd
from scipy.signal import coherence, welch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import networkx as nx
import nolds  # For entropy and Lyapunov exponent calculations
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt


# Function to calculate functional connectivity (correlation and cross-correlation)
def functional_connectivity(time_series):
    n_regions = time_series.shape[1]
    corr_matrix = np.corrcoef(time_series, rowvar=False)

    cross_corr_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            cross_corr_matrix[i, j] = pearsonr(time_series[:, i], time_series[:, j])[0]

    return corr_matrix, cross_corr_matrix


# Function to calculate power spectral density (PSD) and coherence
def spectral_analysis(time_series, fs=1.0):
    n_regions = time_series.shape[1]
    psd_dict = {}
    coherence_matrix = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        f, psd = welch(time_series[:, i], fs)
        psd_dict[f'Region_{i + 1}'] = psd
        for j in range(n_regions):
            if i != j:
                _, coherence_matrix[i, j] = coherence(time_series[:, i], time_series[:, j], fs)

    return psd_dict, coherence_matrix


# Function to compute Sample Entropy
def compute_sample_entropy(time_series):
    entropy_dict = {}
    for i in range(time_series.shape[1]):
        entropy = nolds.sampen(time_series[:, i])
        entropy_dict[f'Region_{i + 1}'] = entropy

    return entropy_dict


# Function to perform Granger Causality
def granger_causality(time_series, maxlag=2):
    n_regions = time_series.shape[1]
    granger_dict = {}

    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                result = grangercausalitytests(time_series[:, [i, j]], maxlag=maxlag, verbose=False)
                granger_dict[f'Region_{i + 1} -> Region_{j + 1}'] = result[maxlag][0]['ssr_ftest'][1]

    return granger_dict


# Function to perform Graph Theoretical Analysis
def graph_theoretical_analysis(connectivity_matrix):
    G = nx.from_numpy_array(connectivity_matrix)
    measures = {
        'Degree Centrality': nx.degree_centrality(G),
        'Clustering Coefficient': nx.clustering(G),
        'Modularity': nx.algorithms.community.modularity(G, nx.algorithms.community.greedy_modularity_communities(G))
    }

    return measures


# Main analysis function
def analyze_time_series(time_series, fs=1.0, maxlag=2):
    # 1. Functional Connectivity
    corr_matrix, cross_corr_matrix = functional_connectivity(time_series)

    # 2. Spectral Analysis
    psd_dict, coherence_matrix = spectral_analysis(time_series, fs=fs)

    # 3. Sample Entropy
    entropy_dict = compute_sample_entropy(time_series)

    # 4. Granger Causality (Effective Connectivity)
    granger_dict = granger_causality(time_series, maxlag=maxlag)

    # 5. Graph Theoretical Analysis
    graph_measures = graph_theoretical_analysis(corr_matrix)

    return {
        'Correlation Matrix': corr_matrix,
        'Cross-Correlation Matrix': cross_corr_matrix,
        'Power Spectral Density': psd_dict,
        'Coherence Matrix': coherence_matrix,
        'Sample Entropy': entropy_dict,
        'Granger Causality': granger_dict,
        'Graph Theoretical Measures': graph_measures
    }