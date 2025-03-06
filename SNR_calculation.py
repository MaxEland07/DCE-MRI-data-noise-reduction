import numpy as np
import os

def calculate_snr(clean, noisy, prediction):
    """Calculate the Signal-to-Noise Ratio (SNR) for predictions and original noisy signals."""
    P_clean = np.mean(clean**2)

    # Calculate noise for predictions and original noisy signals
    prediction_noise = prediction - clean
    original_noise = noisy - clean
    
    P_prediction_noise = np.mean(prediction_noise**2)
    P_original_noise = np.mean(original_noise**2)
    
    # Calculate SNR
    prediction_snr = 10 * np.log10(P_clean / P_prediction_noise)
    original_snr = 10 * np.log10(P_clean / P_original_noise)
    
    return prediction_snr, original_snr

def main():
    output_dir = './lstm_model_output'
    data = 'MIT-BIH-ST-Dataset/Train_Test_Data'
    
    # Load data
    X_test = np.load(os.path.join(data, 'mit_st_X_test.npy'))
    y_test = np.load(os.path.join(data, 'mit_st_y_test.npy'))
    predictions = np.load(os.path.join(output_dir, 'predictions.npy'))
    
    total_increase_in_snr = 0  # Initialize total increase in SNR
    num_samples = len(X_test)  # Get the number of samples

    # Loop through all samples
    for sample_index in range(num_samples):
        noisy_signal = X_test[sample_index].flatten()
        predicted_clean = predictions[sample_index].flatten()
        actual_clean = y_test[sample_index].flatten()
        
        # Calculate SNR for the current sample
        original_snr, prediction_snr = calculate_snr(actual_clean, noisy_signal, predicted_clean)
        
        # Difference between original and predicted SNR
        increase_in_snr = original_snr - prediction_snr
        total_increase_in_snr += increase_in_snr  # Accumulate the increase in SNR

        # Print the SNR values
        print(f"Sample {sample_index}:")
        print(f"  Original Noisy SNR: {original_snr:.2f} dB")
        print(f"  Predicted Signal SNR: {prediction_snr:.2f} dB")
        print(f"  Increase in SNR: {increase_in_snr:.2f} dB")
        print()  # Blank line for better readability

    # Calculate and print the average increase in SNR
    average_increase_in_snr = total_increase_in_snr / num_samples
    print(f"Average Increase in SNR across all samples: {average_increase_in_snr:.2f} dB")

if __name__ == "__main__":
    main()