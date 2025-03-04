import numpy as np
import os
import matplotlib.pyplot as plt

output_dir = './lstm_model_output'
data = 'MIT-BIH-ST-Dataset/Train_Test_Data'

X_test = np.load(os.path.join(data, 'mit_st_X_test.npy'))
y_test = np.load(os.path.join(data, 'mit_st_y_test.npy'))

# Load saved predictions
predictions = np.load(os.path.join(output_dir, 'predictions.npy'))

# Select a sample
sample_index = 0
noisy_signal = X_test[sample_index].flatten()
predicted_clean = predictions[sample_index].flatten()
actual_clean = y_test[sample_index].flatten()

# Plot as above
plt.figure(figsize=(12, 6))
plt.plot(noisy_signal, label='Noisy Signal', alpha=0.5)
plt.plot(predicted_clean, label='Predicted Clean Signal', alpha=0.7)
plt.plot(actual_clean, label='Actual Clean Signal', alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Signal Amplitude')
plt.title('Signal Comparison Using Saved Predictions')
plt.legend()
plt.show()