import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from urllib.parse import urljoin
from sklearn.model_selection import train_test_split
import shutil

# Define directory structure relative to current directory (/kaggle/working/DCE-MRI-data-noise-reduction/)
base_dir = './MIT-BIH-ST-Dataset'  # Relative to current dir
raw_data_dir = os.path.join(base_dir, 'Raw_Data')
processed_data_dir = os.path.join(base_dir, 'Train_Test_Data')

# Clear existing directories to ensure fresh data
for directory in [base_dir, raw_data_dir, processed_data_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

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
    extensions = ['.dat', '.hea', '.atr']
    
    success = True
    for ext in extensions:
        if record_name.endswith('_6'):
            url_name = record_name.replace('_6', '%5F6')
            file_url = urljoin(base_url, f"{url_name}{ext}")
        else:
            file_url = urljoin(base_url, f"{record_name}{ext}")
        file_dest = os.path.join(raw_data_dir, f"{record_name}{ext}")
        if not download_file(file_url, file_dest):
            if ext != '.atr':  # .atr might not exist, that's okay
                success = False
    return success

def download_nstdb():
    """Download the MIT-BIH Noise Stress Test Database"""
    print("Downloading MIT-BIH Noise Stress Test Database...")
    
    # Download clean records
    clean_records = ['118', '119']
    for record in clean_records:
        print(f"Downloading clean record {record}...")
        try:
            wfdb.dl_database('mitdb', raw_data_dir, records=[record])
            print(f"Successfully downloaded {record}")
        except Exception as e:
            print(f"Error downloading {record}: {e}")
    
    # Download noisy records
    noisy_records = [
        '118e24', '118e18', '118e12', '118e06', '118e00', '118e_6',
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
    """Load and prepare data for training"""
    X_clean = []
    X_noisy = {snr: [] for snr in ['24dB', '18dB', '12dB', '6dB', '0dB', '-6dB']}
    
    clean_records = ['118', '119']
    noisy_record_map = {
        '118e24': '24dB', '118e18': '18dB', '118e12': '12dB',
        '118e06': '6dB', '118e00': '0dB', '118e_6': '-6dB',
        '119e24': '24dB', '119e18': '18dB', '119e12': '12dB',
        '119e06': '6dB', '119e00': '0dB', '119e_6': '-6dB'
    }
    
    clean_signals = {}
    for record_name in clean_records:
        try:
            record_path = os.path.join(raw_data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            clean_signals[record_name] = {'signal': record.p_signal[:, 0], 'fs': record.fs}
            print(f"Loaded clean record {record_name}, length: {len(clean_signals[record_name]['signal'])}")
        except Exception as e:
            print(f"Error loading clean record {record_name}: {e}")
    
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
            
            noisy_segment_starts = [5, 9, 13]  # minutes
            segment_duration = 2  # minutes
            for segment_idx, start_min in enumerate(noisy_segment_starts):
                start_sample = int(start_min * 60 * fs)
                end_sample = int((start_min + segment_duration) * 60 * fs)
                if end_sample > len(noisy_signal):
                    print(f"Segment {segment_idx+1} exceeds record length, skipping")
                    continue
                chunk_start = start_sample
                chunk_idx = 0
                while chunk_start + target_length <= end_sample:
                    chunk_end = chunk_start + target_length
                    noisy_segment = noisy_signal[chunk_start:chunk_end]
                    clean_segment = clean_signals[base_record]['signal'][chunk_start:chunk_end]
                    X_noisy[snr_level].append({
                        'segment': noisy_segment, 'record': base_record,
                        'start_idx': chunk_start, 'segment_idx': segment_idx, 'chunk_idx': chunk_idx
                    })
                    X_clean.append({
                        'segment': clean_segment, 'record': base_record,
                        'start_idx': chunk_start, 'segment_idx': segment_idx, 'chunk_idx': chunk_idx,
                        'snr': snr_level
                    })
                    chunk_start += stride
                    chunk_idx += 1
                print(f"Generated {chunk_idx} chunks for {record_name} segment {segment_idx+1}")
        except Exception as e:
            print(f"Error processing noisy record {record_name}: {e}")
    
    # Convert to paired data
    paired_data = []
    for snr in X_noisy:
        for noisy_item in X_noisy[snr]:
            key = f"{noisy_item['record']}_{noisy_item['segment_idx']}_{noisy_item['chunk_idx']}"
            for clean_item in X_clean:
                clean_key = f"{clean_item['record']}_{clean_item['segment_idx']}_{clean_item['chunk_idx']}"
                if key == clean_key:
                    paired_data.append({'clean': clean_item['segment'], 'noisy': noisy_item['segment'], 'snr': snr})
                    break
    
    X_clean_array = np.array([pair['clean'] for pair in paired_data]).reshape(-1, target_length, 1)
    X_noisy_array = np.array([pair['noisy'] for pair in paired_data]).reshape(-1, target_length, 1)
    
    return X_clean_array, X_noisy_array

if __name__ == "__main__":
    # Download data
    download_nstdb()
    
    # Load and prepare data
    X_clean, X_noisy = load_and_prepare_data(target_length=512, stride=256)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, X_clean, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Save files
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