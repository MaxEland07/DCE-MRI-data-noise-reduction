import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Directory paths
output_dir = './lstm_model_output'
data_dir = './MIT-BIH-ST-Dataset/Train_Test_Data'

def download_clean_record(record_name, pn_dir='mitdb'):
    """Download a clean record from the MIT-BIH database on PhysioNet."""
    # Use pn_dir to specify the MIT-BIH Arrhythmia Database on PhysioNet
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    return record.p_signal[:, 0], record.fs  # Return the signal and sampling frequency

def add_noise(signal, noise_level):
    """Add Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_level, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

def normalize_signal(signal, norm_stats):
    """Normalize signal using the same stats as training data."""
    return (signal - norm_stats['noisy_mean']) / norm_stats['noisy_std']

def denormalize_signal(signal, norm_stats):
    """Denormalize signal using the same stats as training data."""
    return signal * norm_stats['clean_std'] + norm_stats['clean_mean']

def segment_signal(signal, segment_length=512):
    """Segment the signal into chunks of specified length."""
    n_samples = len(signal)
    n_segments = n_samples // segment_length
    segmented_signal = signal[:n_segments * segment_length].reshape(n_segments, segment_length, 1)
    return segmented_signal

def define_model():
    """Define the same model architecture as in training."""
    model = Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), 
                                     input_shape=(512, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, 'linear')
    ])
    return model

def calculate_snr(clean, noisy_or_predicted):
    """Calculate the Signal-to-Noise Ratio (SNR)."""
    P_signal = np.mean(clean**2)
    noise = noisy_or_predicted - clean
    P_noise = np.mean(noise**2)
    if P_noise == 0:
        return np.inf
    snr = 10 * np.log10(P_signal / P_noise)
    return snr

def main():
    # Parameters
    record_name = '100'  # Example record from MIT-BIH Arrhythmia Database
    noise_level = 0.2  # Noise level for Gaussian noise
    weights_path = os.path.join(output_dir, 'model.weights.h5')

    # Load normalization stats
    norm_stats = np.load(os.path.join(data_dir, 'normalization_stats.npy'), allow_pickle=True).item()
    print("Loaded normalization statistics.")

    # Step 1: Download the clean record from PhysioNet
    clean_signal, fs = download_clean_record(record_name, pn_dir='mitdb')
    print(f"Downloaded clean record {record_name} with length {len(clean_signal)} samples.")

    # Step 2: Add noise to the clean signal
    noisy_signal = add_noise(clean_signal, noise_level)
    print(f"Added noise to the clean signal.")

    # Step 3: Segment the signals
    clean_signal_segmented = segment_signal(clean_signal)
    noisy_signal_segmented = segment_signal(noisy_signal)
    print(f"Segmented signals into {clean_signal_segmented.shape[0]} chunks of 512 samples.")

    # Step 4: Normalize the noisy signal for prediction
    noisy_signal_normalized = normalize_signal(noisy_signal_segmented, norm_stats)

    # Step 5: Load the trained model
    model = define_model()
    model.load_weights(weights_path)
    print("Model weights loaded successfully.")

    # Step 6: Make predictions
    predicted_signal_normalized = model.predict(noisy_signal_normalized, batch_size=128)
    predicted_signal_denorm = denormalize_signal(predicted_signal_normalized, norm_stats)
    print("Predictions generated.")

    # Step 7: Flatten the segmented signals for comparison
    clean_signal_flat = clean_signal_segmented.flatten()[:len(clean_signal)]
    noisy_signal_flat = noisy_signal_segmented.flatten()[:len(noisy_signal)]
    predicted_signal_flat = predicted_signal_denorm.flatten()[:len(clean_signal)]

    # Step 8: Calculate SNR
    noisy_snr = calculate_snr(clean_signal_flat, noisy_signal_flat)
    predicted_snr = calculate_snr(clean_signal_flat, predicted_signal_flat)
    print(f"Noisy SNR: {noisy_snr:.2f} dB")
    print(f"Predicted SNR: {predicted_snr:.2f} dB")

    # Step 9: Plot the results (first 512 samples for clarity)
    plt.figure(figsize=(12, 6))
    plt.plot(clean_signal_flat[:512], label='Clean Signal', alpha=0.5)
    plt.plot(noisy_signal_flat[:512], label='Noisy Signal', alpha=0.7)
    plt.plot(predicted_signal_flat[:512], label='Predicted Signal', alpha=0.7)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Signal Comparison (First 512 Samples)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'unseen_signal_comparison.png'))
    plt.show()

if __name__ == "__main__":
    main()