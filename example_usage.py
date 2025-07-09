#!/usr/bin/env python3
"""
Example usage script for CrotonePhi: Integrated Information Analysis Pipeline

This script demonstrates how to configure and run the CrotonePhi analysis pipeline.
Modify the paths and parameters according to your data and requirements.

Developer: Dilanjan DK (ddiyabal@uwo.ca)
Development: BrainLab, Western University
Supervisor: Dr. Andrea Soddu
"""

import os
import sys
from Calculate_Phi import main, load_and_process_data, perform_pyphi_analysis

def example_configuration():
    """
    Example configuration for CrotonePhi analysis.
    
    This function demonstrates how to set up the analysis parameters
    and run the pipeline with custom settings.
    """
    
    # =======================
    # Configuration Section
    # =======================
    
    # Analysis parameters
    ANALYSIS_MODE = 3  # Choose: 1 (Regions as nodes), 2 (Frequency as nodes), 3 (Regions within frequency bands)
    OVERWRITE = False  # Set to True to overwrite existing results
    
    # Mode-specific parameters
    NUM_REGIONS_MODE1 = 5  # Required number of regions for Mode 1
    NUM_REGIONS_MODE3 = 5  # Required number of regions for Mode 3
    NUM_FREQUENCIES_MODE3 = 5  # Required number of frequencies for Mode 3
    
    # Data paths - UPDATE THESE FOR YOUR SYSTEM
    dest_dir = '/path/to/your/data/directory'  # Directory containing subject data
    output_dir = '/path/to/your/output/directory'  # Directory for output files
    
    # Subject and network configuration
    subjects = ['subject1', 'subject2']  # Add your subject IDs here
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
    
    return {
        'analysis_mode': ANALYSIS_MODE,
        'overwrite': OVERWRITE,
        'num_regions_mode1': NUM_REGIONS_MODE1,
        'num_regions_mode3': NUM_REGIONS_MODE3,
        'num_frequencies_mode3': NUM_FREQUENCIES_MODE3,
        'dest_dir': dest_dir,
        'output_dir': output_dir,
        'subjects': subjects,
        'network_types': network_types
    }

def run_example_analysis():
    """
    Run the CrotonePhi analysis with example configuration.
    """
    
    print("CrotonePhi: Integrated Information Analysis Pipeline")
    print("=" * 60)
    print("Developer: Dilanjan DK (ddiyabal@uwo.ca)")
    print("Development: BrainLab, Western University")
    print("Supervisor: Dr. Andrea Soddu")
    print("=" * 60)
    
    # Get configuration
    config = example_configuration()
    
    # Validate paths
    if not os.path.exists(config['dest_dir']):
        print(f"ERROR: Data directory does not exist: {config['dest_dir']}")
        print("Please update the 'dest_dir' path in the configuration.")
        return False
    
    if not os.path.exists(config['output_dir']):
        print(f"Creating output directory: {config['output_dir']}")
        os.makedirs(config['output_dir'], exist_ok=True)
    
    # Validate analysis mode
    if config['analysis_mode'] not in [1, 2, 3]:
        print(f"ERROR: Invalid analysis mode: {config['analysis_mode']}")
        print("Please choose 1, 2, or 3.")
        return False
    
    print(f"Analysis Mode: {config['analysis_mode']}")
    print(f"Data Directory: {config['dest_dir']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Subjects: {config['subjects']}")
    print(f"Network Types: {config['network_types']}")
    print("-" * 60)
    
    try:
        # Load and process data
        print("Loading and processing data...")
        dataarrays = load_and_process_data(
            config['dest_dir'], 
            config['subjects'], 
            config['network_types'], 
            config['output_dir']
        )
        
        if not dataarrays:
            print("WARNING: No data arrays were loaded. Check your data paths and file structure.")
            return False
        
        print(f"Successfully loaded {len(dataarrays)} data arrays.")
        
        # Perform PyPhi analysis
        print("Performing PyPhi analysis...")
        perform_pyphi_analysis(
            dataarrays,
            config['output_dir'],
            config['analysis_mode'],
            overwrite=config['overwrite'],
            num_regions_mode1=config['num_regions_mode1'] if config['analysis_mode'] == 1 else None,
            num_regions_mode3=config['num_regions_mode3'] if config['analysis_mode'] == 3 else None,
            num_frequencies_mode3=config['num_frequencies_mode3'] if config['analysis_mode'] == 3 else None
        )
        
        print("Analysis completed successfully!")
        print(f"Results saved to: {config['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Analysis failed with exception: {e}")
        return False

def print_usage_instructions():
    """
    Print usage instructions for the CrotonePhi pipeline.
    """
    
    print("\n" + "=" * 60)
    print("CrotonePhi Usage Instructions")
    print("=" * 60)
    
    print("\n1. CONFIGURATION:")
    print("   - Update paths in example_usage.py or Calculate_Phi.py")
    print("   - Set your data directory and output directory")
    print("   - Configure subjects and network types")
    
    print("\n2. DATA REQUIREMENTS:")
    print("   - MATLAB (.mat) files with 'TF' variable")
    print("   - Expected shape: (n_regions, time_points, frequencies)")
    print("   - Optional 'Freqs' variable for frequency labels")
    
    print("\n3. DIRECTORY STRUCTURE:")
    print("   data_directory/")
    print("   ├── subject1/")
    print("   │   └── Task1/")
    print("   │       ├── network_type1/")
    print("   │       │   └── data.mat")
    print("   │       └── network_type2/")
    print("   │           └── data.mat")
    print("   └── subject2/")
    print("       └── Task1/")
    print("           └── ...")
    
    print("\n4. ANALYSIS MODES:")
    print("   Mode 1: Regions as nodes (sum across frequencies)")
    print("   Mode 2: Frequency as nodes (sum across regions)")
    print("   Mode 3: Regions as nodes within each frequency band")
    
    print("\n5. OUTPUT FILES:")
    print("   - phi.csv: Integrated information values")
    print("   - tpm_sbs.csv: State-by-state transition probability matrix")
    print("   - tpm_sbn.csv: State-by-node transition probability matrix")
    print("   - subject_network_type.nc: Processed NetCDF data")
    
    print("\n6. CONTACT:")
    print("   Developer: Dilanjan DK (ddiyabal@uwo.ca)")
    print("   Institution: BrainLab, Western University")
    print("   Supervisor: Dr. Andrea Soddu")

if __name__ == "__main__":
    """
    Main execution block.
    """
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print_usage_instructions()
    else:
        success = run_example_analysis()
        if not success:
            print("\nFor help, run: python example_usage.py --help")
            sys.exit(1) 