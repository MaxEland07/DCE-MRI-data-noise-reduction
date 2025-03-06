import numpy as np
import os

def calculate_snr(clean, noisy_or_predicted):
    """Calculate the Signal-to-Noise Ratio (SNR).

    Args:
        clean (np.ndarray): The clean signal.
        noisy_or_predicted (np.ndarray): The noisy or predicted signal.

    Returns:
        float: The SNR value in dB.
    """
    P_signal = np.mean(clean**2)
    noise = noisy_or_predicted - clean
    P_noise = np.mean(noise**2)
    
    if P_noise == 0:
        return np.inf  # Handle the case where there is no noise
    
    snr = 10 * np.log10(P_signal / P_noise)
    return snr

def main():
    output_dir = './lstm_model_output'
    data = 'MIT-BIH-ST-Dataset/Train_Test_Data'
    
    # Load data
    X_test = np.load(os.path.join(data, 'mit_st_X_test.npy'))
    y_test = np.load(os.path.join(data, 'mit_st_y_test.npy'))
    predictions = np.load(os.path.join(output_dir, 'predictions.npy'))
    
    num_samples = len(X_test)

    for sample_index in range(num_samples):
        noisy_signal = X_test[sample_index].flatten()
        predicted_signal = predictions[sample_index].flatten()
        clean_signal = y_test[sample_index].flatten()
        
        # Calculate SNR for noisy and predicted signals
        noisy_snr = calculate_snr(clean_signal, noisy_signal)
        predicted_snr = calculate_snr(clean_signal, predicted_signal)
        
        # Calculate the increase in SNR
        increase_in_snr = predicted_snr - noisy_snr
        
        print(f"Sample {sample_index}:")
        print(f"  Noisy SNR: {noisy_snr:.2f} dB")
        print(f"  Predicted SNR: {predicted_snr:.2f} dB")
        print(f"  Increase in SNR: {increase_in_snr:.2f} dB")
        print()

if __name__ == "__main__":
    main()
