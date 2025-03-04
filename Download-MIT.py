import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from urllib.parse import urljoin

# Create directory structure
base_dir = '/content/DCE-MRI-data-noise-reduction/MIT-BIH-ST-Dataset'
raw_data_dir = os.path.join(base_dir, 'Raw_Data')
processed_data_dir = os.path.join(base_dir, 'Train_Test_Data')

# Create directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

def download_file(url, dest_path):
    """Download a file from a URL to a destination path"""
    print(f"Downloading {url} to {dest_path}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download {url}, status code {r.status_code}")
        return False

def download_nstdb_record(record_name):
    """Download a specific record from the MIT-BIH Noise Stress Test Database"""
    base_url = "https://physionet.org/files/nstdb/1.0.0/"
    
    # Files to download for each record
    extensions = ['.dat', '.hea', '.atr']
    
    success = True
    for ext in extensions:
        # Handle special case for negative SNR records
        if record_name.endswith('_6'):
            # URL-encode the underscore
            url_name = record_name.replace('_6', '%5F6')
            file_url = urljoin(base_url, f"{url_name}{ext}")
        else:
            file_url = urljoin(base_url, f"{record_name}{ext}")
        
        file_dest = os.path.join(raw_data_dir, f"{record_name}{ext}")
        
        # Skip download if file already exists
        if os.path.exists(file_dest):
            print(f"File {file_dest} already exists, skipping download")
            continue
        
        if not download_file(file_url, file_dest):
            # If .atr file fails, it might not exist for this record, that's ok
            if ext != '.atr':
                success = False
    
    return success

def download_nstdb():
    """Download the MIT-BIH Noise Stress Test Database"""
    print("Downloading MIT-BIH Noise Stress Test Database...")
    
    # Download clean records from MIT-BIH Arrhythmia Database
    clean_records = ['118', '119']
    for record in clean_records:
        print(f"Downloading clean record {record}...")
        try:
            wfdb.dl_database('mitdb', os.path.join(raw_data_dir), records=[record])
            print(f"Successfully downloaded {record}")
        except Exception as e:
            print(f"Error downloading {record}: {e}")
    
    # Download noisy records from NSTDB - using correct naming convention from documentation
    # SNR values: 24dB, 18dB, 12dB, 6dB, 0dB, -6dB
    noisy_records = [
        # 118 series
        '118e24', '118e18', '118e12', '118e06', '118e00', '118e_6',
        # 119 series
        '119e24', '119e18', '119e12', '119e06', '119e00', '119e_6'
    ]
    
    for record in noisy_records:
        print(f"Downloading noisy record {record}...")
        success = download_nstdb_record(record)
        if success:
            print(f"Successfully downloaded {record}")
        else:
            print(f"Failed to download some files for {record}")
    
    print("Download complete!")

def load_and_prepare_data(target_length=512, stride=256):
    """
    Load data from MIT-BIH Noise Stress Test Database and prepare it for training
    Returns X_clean (clean ECG signals) and X_noisy (noisy ECG signals)
    
    Args:
        target_length: Length of each segment (default: 512)
        stride: Step size between segments (default: 256)
    """
    X_clean = []
    X_noisy = {}  # Use dictionary to organize by SNR
    
    # Initialize dictionary for each SNR level
    for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']:
        X_noisy[snr] = []
    
    # Clean records
    clean_records = ['118', '119']
    
    # Noisy records with different SNR levels
    noisy_record_map = {
        '118e24': '24dB', '118e18': '18dB', '118e12': '12dB', 
        '118e06': '6dB', '118e00': '0dB', '118e_6': '-6dB',
        '119e24': '24dB', '119e18': '18dB', '119e12': '12dB', 
        '119e06': '6dB', '119e00': '0dB', '119e_6': '-6dB'
    }
    
    clean_signals = {}
    
    # Load clean records
    for record_name in clean_records:
        try:
            record_path = os.path.join(raw_data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            clean_signals[record_name] = {
                'signal': record.p_signal[:, 0],
                'fs': record.fs
            }
            print(f"Loaded clean record {record_name}, length: {len(clean_signals[record_name]['signal'])}")
        except Exception as e:
            print(f"Error loading clean record {record_name}: {e}")
    
    # Load and process noisy records
    for record_name, snr_level in noisy_record_map.items():
        try:
            base_record = record_name[:3]
            
            if base_record not in clean_signals:
                print(f"Warning: Clean record {base_record} not found, skipping {record_name}")
                continue
                
            record_path = os.path.join(raw_data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            fs = record.fs
            
            noisy_signal = record.p_signal[:, 0]
            print(f"Loaded noisy record {record_name}, length: {len(noisy_signal)}")
            
            # Define noisy segment start times (in minutes)
            noisy_segment_starts = [5, 9, 13]  # minutes
            segment_duration = 2  # minutes
            
            for segment_idx, start_min in enumerate(noisy_segment_starts):
                start_sample = int(start_min * 60 * fs)
                end_sample = int((start_min + segment_duration) * 60 * fs)
                
                if end_sample > len(noisy_signal):
                    print(f"Segment {segment_idx+1} exceeds record length, skipping")
                    continue
                
                # Modified chunk generation with stride
                chunk_start = start_sample
                chunk_idx = 0
                
                while chunk_start + target_length <= end_sample:
                    chunk_end = chunk_start + target_length
                    
                    noisy_segment = noisy_signal[chunk_start:chunk_end]
                    clean_segment = clean_signals[base_record]['signal'][chunk_start:chunk_end]
                    
                    X_noisy[snr_level].append({
                        'segment': noisy_segment,
                        'record': base_record,
                        'start_idx': chunk_start,
                        'segment_idx': segment_idx,
                        'chunk_idx': chunk_idx
                    })
                    
                    X_clean.append({
                        'segment': clean_segment,
                        'record': base_record,
                        'start_idx': chunk_start,
                        'segment_idx': segment_idx,
                        'chunk_idx': chunk_idx,
                        'snr': snr_level
                    })
                    
                    chunk_start += stride
                    chunk_idx += 1
                
                print(f"Generated {chunk_idx} chunks for {record_name} segment {segment_idx+1}")
        
        except Exception as e:
            print(f"Error processing noisy record {record_name}: {e}")
    
    # Print summary
    print("\nData collection summary:")
    total_clean = len(X_clean)
    print(f"Total segments collected: {total_clean}")
    for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']:
        if X_noisy[snr]:
            print(f"SNR {snr}: {len(X_noisy[snr])} segments")
        else:
            print(f"SNR {snr}: No segments found")    
    
    # Create visualization examples that show the same segment across all SNR levels
    visual_examples = []
    
    # First convert to a more convenient format for finding matching segments
    segment_key_to_clean = {}
    segment_key_to_noisy = {snr: {} for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']}
    
    for item in X_clean:
        key = f"{item['record']}_{item['segment_idx']}_{item['chunk_idx']}"
        segment_key_to_clean[key] = item
    
    for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']:
        for item in X_noisy[snr]:
            key = f"{item['record']}_{item['segment_idx']}_{item['chunk_idx']}"
            segment_key_to_noisy[snr][key] = item
    
    # Find segments that exist across all SNR levels
    all_segment_keys = set(segment_key_to_clean.keys())
    for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']:
        all_segment_keys &= set(segment_key_to_noisy[snr].keys())
    
    print(f"Found {len(all_segment_keys)} segments that exist across all SNR levels")
    
    # Create visual examples
    for key in list(all_segment_keys)[:10]:  # Take first 10 examples
        example = {
            'clean': segment_key_to_clean[key],
            'noisy': {snr: segment_key_to_noisy[snr][key] for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']}
        }
        visual_examples.append(example)
    
    # Now organize the data for training
    # We'll create a paired dataset where each noisy segment is matched with its clean counterpart
    paired_data = []
    for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']:
        for noisy_item in X_noisy[snr]:
            key = f"{noisy_item['record']}_{noisy_item['segment_idx']}_{noisy_item['chunk_idx']}"
            if key in segment_key_to_clean:
                paired_data.append({
                    'clean': segment_key_to_clean[key]['segment'],
                    'noisy': noisy_item['segment'],
                    'snr': snr
                })
    
    print(f"Created {len(paired_data)} paired samples")
    
    X_clean_array = np.array([pair['clean'] for pair in paired_data])
    X_noisy_array = np.array([pair['noisy'] for pair in paired_data])
    
    X_clean_array = X_clean_array.reshape(X_clean_array.shape[0], X_clean_array.shape[1], 1)
    X_noisy_array = X_noisy_array.reshape(X_noisy_array.shape[0], X_noisy_array.shape[1], 1)
    
    X_noisy_by_level = {}
    for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']:
        snr_samples = [pair['noisy'] for pair in paired_data if pair['snr'] == snr]
        if snr_samples:
            X_noisy_by_level[snr] = np.array(snr_samples).reshape(-1, target_length, 1)
    
    return X_clean_array, X_noisy_array, X_noisy_by_level, visual_examples
    
if __name__ == "__main__":
    # Download the database
    download_nstdb()
    
    # Load and prepare data
    X_clean, X_noisy, X_noisy_by_level, visual_examples = load_and_prepare_data(
      target_length=512,
      stride=256
    )
    
    # Create a plots directory
    plots_dir = os.path.join(base_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot one example for each SNR level from matched examples
    if visual_examples:
        for example_idx, example in enumerate(visual_examples[:5]):
            # Create a figure to show all SNR levels for one segment
            plt.figure(figsize=(15, 14))
            
            # Plot the clean signal first
            plt.subplot(7, 1, 1)
            plt.plot(example['clean']['segment'])
            plt.title(f"Clean ECG Signal (Record {example['clean']['record']})")
            
            # Plot each SNR level from highest to lowest
            for i, snr in enumerate(['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']):
                plt.subplot(7, 1, i+2)
                plt.plot(example['noisy'][snr]['segment'])
                plt.title(f"Noisy ECG Signal - SNR {snr}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'noise_comparison_example_{example_idx+1}.png'))
            plt.close()
            
            # Also create individual comparisons for each SNR level
            for i, snr in enumerate(['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']):
                plt.figure(figsize=(12, 6))
                
                plt.subplot(2, 1, 1)
                plt.plot(example['noisy'][snr]['segment'])
                plt.title(f'Example {example_idx+1}: Noisy ECG Signal (SNR {snr})')
                
                plt.subplot(2, 1, 2)
                plt.plot(example['clean']['segment'])
                plt.title(f'Example {example_idx+1}: Clean ECG Signal')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'example_{example_idx+1}_snr_{snr}.png'))
                plt.close()
    
    # Split into training and testing sets
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, X_clean, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Save the dataset to the processed data directory
    np.save(os.path.join(processed_data_dir, 'mit_st_X_train.npy'), X_train)
    np.save(os.path.join(processed_data_dir, 'mit_st_y_train.npy'), y_train)
    np.save(os.path.join(processed_data_dir, 'mit_st_X_test.npy'), X_test)
    np.save(os.path.join(processed_data_dir, 'mit_st_y_test.npy'), y_test)
    
    print("Data preparation complete!")
    print(f"Files saved in {processed_data_dir}:")
    print("- mit_st_X_train.npy")
    print("- mit_st_y_train.npy") 
    print("- mit_st_X_test.npy")
    print("- mit_st_y_test.npy")
    print(f"Example plots saved in {plots_dir}")

    #X_train shape: (4809, 512, 1)