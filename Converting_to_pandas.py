import os
import pandas as pd
from scipy.io import loadmat
import numpy as np

# Define the base destination directory - UPDATE THIS PATH FOR YOUR SYSTEM
dest_dir = r'/path/to/your/data/directory'  # Update this path

# Define the subjects and their network types
subjects = ['']  # Add your subjects here
network_types = [
    'frontoparietal',
    'Salience',
    'Visual',
    'Auditory',
    'cingulooperc',
    'Default',
    'retrosplinialtemporal',
    'smmouth',
    'cinguloparietal'
]

# Dictionary to store the dataframes
dataframes = {}

# Iterate through each subject and network type
for subject in subjects:
    subject_dir = os.path.join(dest_dir, subject, 'Task1')

    for network_type in network_types:
        network_dir = os.path.join(subject_dir, network_type)

        # Assuming there's only one .mat file in each directory
        for file in os.listdir(network_dir):
            if file.endswith('.mat'):
                file_path = os.path.join(network_dir, file)
                print(f'Loading {file_path}')

                # Load the .mat file
                mat_data = loadmat(file_path)

                # Load the relevant data
                tf_data = mat_data['TF']  # Shape is (5, 3819, 6)

                # Generate labels for the regions (1 to 5)
                region_labels = [f'Region_{i + 1}' for i in range(tf_data.shape[0])]
                time_labels = np.arange(1, tf_data.shape[1] + 1)

                # Generate frequency band labels (1 to 6)
                freq_labels = [freq[0][0] for freq in mat_data['Freqs']]

                # Create a MultiIndex for the rows
                index = pd.MultiIndex.from_product([region_labels, time_labels], names=["Region", "Time"])

                # Reshape the TF data to fit into a DataFrame
                tf_reshaped = tf_data.reshape(-1, tf_data.shape[2])

                # Create the DataFrame
                df_tf = pd.DataFrame(tf_reshaped, index=index, columns=freq_labels)

                # Save the DataFrame to the corresponding directory as a pickle file
                output_file = os.path.join(network_dir, f'{subject}_{network_type}.pkl')
                df_tf.to_pickle(output_file)
                print(f'Saved DataFrame to {output_file}')

                # Store the DataFrame in the dictionary (if needed later)
                dataframes[(subject, network_type)] = df_tf


