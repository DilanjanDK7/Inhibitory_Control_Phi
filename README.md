# Integrated Information Analysis Pipeline

A comprehensive Python pipeline for analyzing integrated information (Phi) in brain network data using the PyPhi framework. This tool is designed for processing time-frequency data from brain regions and computing integrated information measures across different analysis modes.

## Author Information

- **Developer**: Dilanjan DK (ddiyabal@uwo.ca)
- **Development**: BrainLab, Western University
- **Supervisor**: Dr. Andrea Soddu
- **Institution**: Western University

## Overview

CrotonePhi provides a complete workflow for:
1. Loading and preprocessing time-frequency brain data from MATLAB (.mat) files
2. Converting data to NetCDF format for efficient storage and processing
3. Computing integrated information (Phi) using the PyPhi framework
4. Supporting multiple analysis modes for different network configurations
5. Generating comprehensive output files for further analysis

## Features

### Analysis Modes
- **Mode 1**: Regions as nodes (sum of all frequencies)
- **Mode 2**: Frequency as nodes (sum of all regions)  
- **Mode 3**: Regions as nodes within each frequency band

### Data Processing
- Automatic discretization of time series data to binary states
- Transition Probability Matrix (TPM) construction
- Support for multiple frequency bands (delta, theta, alpha, beta, gamma1, gamma2)
- Flexible network type configurations

### Output Generation
- Phi values for each analysis configuration
- Transition Probability Matrices (state-by-state and state-by-node formats)
- Comprehensive logging and progress tracking
- NetCDF data storage for processed time-frequency data

## Installation

### Prerequisites
- Python 3.7+
- PyPhi framework
- Required Python packages (see requirements below)

### Dependencies
```bash
pip install numpy pandas xarray scipy pyphi matplotlib
```

### Core Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `xarray`: Multi-dimensional data handling
- `scipy`: Scientific computing
- `pyphi`: Integrated information computation
- `matplotlib`: Visualization (optional)

## Usage

### Configuration

1. **Update Paths**: Modify the data and output directories in `Calculate_Phi.py`:
   ```python
   dest_dir = '/path/to/your/data/directory'  # Your data directory
   output_dir = '/path/to/your/output/directory'  # Your output directory
   ```

2. **Set Analysis Parameters**:
   ```python
   ANALYSIS_MODE = 3  # Choose 1, 2, or 3
   OVERWRITE = False  # Set to True to overwrite existing results
   ```

3. **Configure Network Types**: Modify the `network_types` list as needed:
   ```python
   network_types = [
       'frontoparietal', 'Salience', 'Visual', 'Auditory',
       'cingulooperc', 'Default', 'retrosplinialtemporal',
       'smmouth', 'smhand', 'cinguloparietal', 'Dorsalattn'
   ]
   ```

### Running the Analysis

```bash
python Calculate_Phi.py
```

### Data Format Requirements

Input data should be MATLAB (.mat) files containing:
- `TF`: Time-frequency data array (shape: n_regions × T × F)
- `Freqs`: Frequency labels (optional)

Expected directory structure:
```
data_directory/
├── subject1/
│   └── Task1/
│       ├── network_type1/
│       │   └── data.mat
│       └── network_type2/
│           └── data.mat
└── subject2/
    └── Task1/
        └── ...
```

## File Structure

```
CrotonePhi/
├── Calculate_Phi.py          # Main analysis pipeline
├── Converting_to_pandas.py   # Data conversion utilities
├── Utils/
│   ├── Core_Phi.py          # Core PyPhi analysis functions
│   ├── Core_Phi_Backup.py   # Backup of core functions
│   ├── Core_Phi_only_phi.py # Phi-specific computations
│   ├── Core.py              # General analysis utilities
│   └── Utils_Lab.py         # Laboratory utilities
└── README.md                # This file
```

## Output Structure

The pipeline generates the following output structure:

```
output_directory/
├── Data/
│   └── subject/
│       └── network_type/
│           └── subject_network_type.nc
└── Phi_Results/
    └── subject/
        └── network_type/
            ├── Regions_Sum_Frequencies/     # Mode 1
            │   ├── phi.csv
            │   ├── tpm_sbs.csv
            │   └── tpm_sbn.csv
            ├── Frequencies_Sum_Regions/     # Mode 2
            │   ├── phi.csv
            │   ├── tpm_sbs.csv
            │   └── tpm_sbn.csv
            └── Frequency_band/              # Mode 3
                ├── phi.csv
                ├── tpm_sbs.csv
                └── tpm_sbn.csv
```

## Key Functions

### Core Analysis Functions
- `load_and_process_data()`: Loads and preprocesses MATLAB data
- `perform_pyphi_analysis()`: Executes PyPhi analysis with multiple modes
- `discretize_time_series_binary()`: Converts time series to binary states
- `construct_tpm()`: Builds Transition Probability Matrices
- `to_calculate_mean_phi()`: Computes integrated information measures

### Utility Functions
- `tpm_le()`: Legacy TPM construction function
- `binary_ts()`: Time series binarization
- `get_envelope()`: Signal envelope computation
- `distance_arrays()`: Array distance calculations

## Analysis Modes Explained

### Mode 1: Regions as Nodes
- Sums across all frequency bands for each brain region
- Creates TPMs where each node represents a brain region
- Useful for studying regional interactions independent of frequency

### Mode 2: Frequency as Nodes  
- Sums across all brain regions for each frequency band
- Creates TPMs where each node represents a frequency band
- Useful for studying frequency-specific dynamics

### Mode 3: Regions within Frequency Bands
- Analyzes each frequency band separately with regions as nodes
- Creates separate TPMs for each frequency band
- Useful for studying frequency-specific regional interactions

## Configuration Options

### Analysis Parameters
- `ANALYSIS_MODE`: Analysis mode selection (1, 2, or 3)
- `OVERWRITE`: Whether to overwrite existing results
- `NUM_REGIONS_MODE1`: Required number of regions for Mode 1
- `NUM_REGIONS_MODE3`: Required number of regions for Mode 3
- `NUM_FREQUENCIES_MODE3`: Required number of frequencies for Mode 3

### Data Processing Options
- Frequency band filtering (delta, theta, alpha, beta, gamma1, gamma2)
- Automatic NaN handling with interpolation
- Flexible discretization thresholds

## Troubleshooting

### Common Issues
1. **Path Errors**: Ensure all directory paths are correctly set
2. **Memory Issues**: Large datasets may require increased memory allocation
3. **PyPhi Errors**: Check PyPhi installation and version compatibility
4. **Data Format**: Verify MATLAB file structure and variable names

### Performance Optimization
- Use `OVERWRITE=False` to skip existing analyses
- Adjust `NUM_REGIONS_MODE1/3` to limit processing scope
- Consider data subsetting for initial testing

## Citations

### CrotonePhi Citation
If you use this software in your research, please cite:

```
CrotonePhi: Integrated Information Analysis Pipeline
Developer: Dilanjan DK (ddiyabal@uwo.ca)
Development: BrainLab, Western University
Supervisor: Dr. Andrea Soddu
```

### PyPhi Citation
This software uses the PyPhi framework for integrated information computation. When using CrotonePhi, please also cite PyPhi:

```
Mayner, W. G. P., Marshall, W., Albantakis, L., Findlay, G., Marchman, R., & Tononi, G. (2018). 
PyPhi: A toolbox for integrated information theory. 
PLoS computational biology, 14(7), e1006343.
```

**BibTeX:**
```bibtex
@article{mayner2018pyphi,
  title={PyPhi: A toolbox for integrated information theory},
  author={Mayner, William GP and Marshall, William and Albantakis, Larissa and Findlay, Graham and Marchman, Robert and Tononi, Giulio},
  journal={PLoS computational biology},
  volume={14},
  number={7},
  pages={e1006343},
  year={2018},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

### Integrated Information Theory Citation
For the theoretical foundation, please also cite:

```
Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). 
Integrated information theory: from consciousness to its physical substrate. 
Nature Reviews Neuroscience, 17(7), 450-461.
```

**BibTeX:**
```bibtex
@article{tononi2016integrated,
  title={Integrated information theory: from consciousness to its physical substrate},
  author={Tononi, Giulio and Boly, Melanie and Massimini, Marcello and Koch, Christof},
  journal={Nature Reviews Neuroscience},
  volume={17},
  number={7},
  pages={450--461},
  year={2016},
  publisher={Nature Publishing Group}
}
```

## License

This software is provided for research purposes. Please contact the developer for commercial use.

## Contact

For questions, bug reports, or feature requests, please contact:
- **Developer**: Dilanjan DK (ddiyabal@uwo.ca)
- **Institution**: BrainLab, Western University
- **Supervisor**: Dr. Andrea Soddu

## Acknowledgments

- PyPhi development team for the integrated information framework
- BrainLab at Western University for computational resources
- Dr. Andrea Soddu for supervision and guidance 