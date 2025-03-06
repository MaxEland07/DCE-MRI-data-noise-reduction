import numpy as np
import os

def calculate_snr(signal, prediction):
    """Calculate the Signal-to-Noise Ratio (SNR)."""
    P_signal = np.mean(signal**2)
    noise = signal - prediction
    P_noise = np.mean(noise**2)
    snr = 10 * np.log10(P_signal / P_noise)
    return snr

def main():
    output_dir = './lstm_model_output'
    data = 'MIT-BIH-ST-Dataset/Train_Test_Data'
    
    # Load data
    X_test = np.load(os.path.join(data, 'mit_st_X_test.npy'))
    y_test = np.load(os.path.join(data, 'mit_st_y_test.npy'))
    predictions = np.load(os.path.join(output_dir, 'predictions.npy'))
    
    # Select a sample
    sample_index = 0
    noisy_signal = X_test[sample_index].flatten()
    predicted_clean = predictions[sample_index].flatten()
    actual_clean = y_test[sample_index].flatten()
    
    # Calculate SNR
    snr_value = calculate_snr(actual_clean, predicted_clean)
    print(f"SNR for sample {sample_index}: {snr_value} dB")

if __name__ == "__main__":
    main()