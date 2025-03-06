import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from urllib.parse import urljoin
from sklearn.model_selection import train_test_split
import shutil
from scipy.signal import butter, filtfilt

# Define directory structure
base_dir = './MIT-BIH-ST-Dataset'
raw_data_dir = os.path.join(base_dir, 'Raw_Data')
processed_data_dir = os.path.join(base_dir, 'Train_Test_Data')

# Clear existing directories
for directory in [base_dir, raw_data_dir, processed_data_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def download_nstdb():
    """Download MIT-BIH Arrhythmia Database records."""
    print("Downloading MIT-BIH Database records...")
    records = ['100', '101', '102', '103', '104', '105', '106', '107', '108']  # 9 records
    for record in records:
        try:
            wfdb.dl_database('mitdb', raw_data_dir, records=[record])
            print(f"Successfully downloaded {record}")
        except Exception as e:
            print(f"Error downloading {record}: {e}")

def add_artificial_noise(signal, snr_db, noise_types, weights):
    """Add a mixture of artificial noises to a clean signal."""
    if len(weights) != len(noise_types) or not np.isclose(sum(weights), 1):
        raise ValueError("Weights must match noise_types length and sum to 1")
    
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    total_noise = np.zeros(len(signal))
    
    for noise_type, weight in zip(noise_types, weights):
        if noise_type == 'gaussian':
            noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        elif noise_type == 'pink':
            white_noise = np.random.normal(0, 1, len(signal))
            noise_fft = np.fft.rfft(white_noise)
            f = np.fft.rfftfreq(len(signal))
            f[0] = f[1]
            noise_fft = noise_fft / np.sqrt(f)
            noise = np.fft.irfft(noise_fft)
            noise = noise * np.sqrt(noise_power / np.mean(noise**2))
        elif noise_type == 'powerline':
            t = np.arange(len(signal))
            noise = (np.sin(2 * np.pi * 50 * t / len(signal)) +
                     0.5 * np.sin(2 * np.pi * 100 * t / len(signal)) +
                     0.3 * np.sin(2 * np.pi * 60 * t / len(signal)))
            noise = noise * np.sqrt(noise_power / np.mean(noise**2))
        elif noise_type == 'baseline':
            t = np.arange(len(signal))
            noise = (np.sin(2 * np.pi * 0.1 * t / len(signal)) +
                     0.5 * np.sin(2 * np.pi * 0.3 * t / len(signal)) +
                     0.2 * np.sin(2 * np.pi * 0.5 * t / len(signal)))
            noise = noise * np.sqrt(noise_power / np.mean(noise**2))
        elif noise_type == 'muscle':
            white_noise = np.random.normal(0, 1, len(signal))
            b, a = butter(4, 0.1, 'highpass')
            noise = filtfilt(b, a, white_noise)
            noise = noise * np.sqrt(noise_power / np.mean(noise**2))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        total_noise += noise * weight
    
    return signal + total_noise

def load_and_prepare_data(target_length=512):
    """Load and prepare data with mixed noise, no sliding window."""
    paired_data = []
    
    records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109']  # 9 records
    snr_levels = {'24dB': 24, '18dB': 18, '12dB': 12, '6dB': 6, '0dB': 0, '-6dB': -6}
    
    noise_mixes = {
        '24dB': (['powerline', 'baseline'], [0.6, 0.4]),
        '18dB': (['gaussian', 'muscle'], [0.5, 0.5]),
        '12dB': (['pink', 'powerline'], [0.7, 0.3]),
        '6dB': (['baseline', 'muscle'], [0.6, 0.4]),
        '0dB': (['gaussian', 'powerline', 'baseline'], [0.4, 0.3, 0.3]),
        '-6dB': (['pink', 'muscle', 'powerline'], [0.5, 0.3, 0.2])
    }
    
    clean_signals = {}
    for record_name in records:
        try:
            record_path = os.path.join(raw_data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0]
            if signal.size == 0:
                print(f"Skipping empty record {record_name}")
                continue
            clean_signals[record_name] = {'signal': signal, 'fs': record.fs}
            print(f"Loaded clean record {record_name}, length: {len(signal)}")
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
    
    for record_name, signal_info in clean_signals.items():
        clean_signal = signal_info['signal']
        fs = signal_info['fs']
        segment_samples = int(2 * 60 * fs)  # 2 minutes
        
        n_segments = len(clean_signal) // segment_samples
        for segment_idx in range(n_segments):
            start_sample = segment_idx * segment_samples
            end_sample = start_sample + segment_samples
            if end_sample > len(clean_signal):
                continue
            
            clean_segment = clean_signal[start_sample:end_sample]
            
            for snr_label, snr_db in snr_levels.items():
                noise_types, weights = noise_mixes[snr_label]
                noisy_segment = add_artificial_noise(clean_segment, snr_db, noise_types, weights)
                
                chunk_start = 0
                chunk_idx = 0
                while chunk_start + target_length <= len(clean_segment):
                    chunk_end = chunk_start + target_length
                    clean_chunk = clean_segment[chunk_start:chunk_end]
                    noisy_chunk = noisy_segment[chunk_start:chunk_end]
                    
                    paired_data.append({
                        'clean': clean_chunk,
                        'noisy': noisy_chunk,
                        'snr': snr_label,
                        'noise_mix': '+'.join(noise_types),
                        'record': record_name
                    })
                    
                    chunk_start += target_length  # No overlap, stride = target_length
                    chunk_idx += 1
                
                print(f"Generated {chunk_idx} chunks for {record_name} segment {segment_idx+1} "
                      f"with {noise_mixes[snr_label][0]} at {snr_label}")
    
    return paired_data

def visualize_sample(paired_data, num_samples=3):
    """Visualize clean vs noisy signals for a few samples."""
    sampled_data = paired_data[:num_samples]
    for i, pair in enumerate(sampled_data):
        plt.figure(figsize=(12, 4))
        plt.plot(pair['clean'], label='Clean', color='blue')
        plt.plot(pair['noisy'], label=f"Noisy ({pair['snr']}, {pair['noise_mix']})", color='red', alpha=0.7)
        plt.title(f"Sample {i+1}: Clean vs Noisy Signal")
        plt.legend()
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()

def create_stratified_split(paired_data, test_size=0.2, random_state=42):
    """Create a stratified split by record."""
    unique_records = list(set([pair['record'] for pair in paired_data]))
    train_records, test_records = train_test_split(unique_records, test_size=test_size, random_state=random_state)
    train_indices = [i for i, pair in enumerate(paired_data) if pair['record'] in train_records]
    test_indices = [i for i, pair in enumerate(paired_data) if pair['record'] in test_records]
    return train_indices, test_indices

if __name__ == "__main__":
    # Download data
    download_nstdb()
    
    # Load and prepare data
    paired_data = load_and_prepare_data(target_length=512)
    
    # Visualize a few samples
    #visualize_sample(paired_data)
    
    # Convert to arrays
    X_clean_array = np.array([pair['clean'] for pair in paired_data]).reshape(-1, 512, 1)
    X_noisy_array = np.array([pair['noisy'] for pair in paired_data]).reshape(-1, 512, 1)
    
    # Normalize
    mean_clean = np.mean(X_clean_array)
    std_clean = np.std(X_clean_array)
    X_clean_array = (X_clean_array - mean_clean) / std_clean
    X_noisy_array = (X_noisy_array - mean_clean) / std_clean
    
    # Stratified split
    train_indices, test_indices = create_stratified_split(paired_data)
    X_train = X_noisy_array[train_indices]
    X_test = X_noisy_array[test_indices]
    y_train = X_clean_array[train_indices]
    y_test = X_clean_array[test_indices]
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Save files
    np.save(os.path.join(processed_data_dir, 'mit_st_X_train.npy'), X_train)
    np.save(os.path.join(processed_data_dir, 'mit_st_y_train.npy'), y_train)
    np.save(os.path.join(processed_data_dir, 'mit_st_X_test.npy'), X_test)
    np.save(os.path.join(processed_data_dir, 'mit_st_y_test.npy'), y_test)
    np.save(os.path.join(processed_data_dir, 'normalization_stats.npy'), {'mean': mean_clean, 'std': std_clean})
    
    print("Data preparation complete!")
    print(f"Files saved in {processed_data_dir}:")
    print("- mit_st_X_train.npy")
    print("- mit_st_y_train.npy")
    print("- mit_st_X_test.npy")
    print("- mit_st_y_test.npy")
    print("- normalization_stats.npy")