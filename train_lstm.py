import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Load data
data_dir = './MIT-BIH-ST-Dataset/Train_Test_Data'
X_train = np.load(os.path.join(data_dir, 'mit_st_X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'mit_st_y_train.npy'))
X_test = np.load(os.path.join(data_dir, 'mit_st_X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'mit_st_y_test.npy'))

# Load normalization statistics (if available)
norm_stats_path = os.path.join(data_dir, 'normalization_stats.npy')
if os.path.exists(norm_stats_path):
    print("Loading pre-computed normalization statistics...")
    norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
else:
    print("No normalization statistics found. Data is assumed to be already normalized.")
    # Create empty stats for later use
    norm_stats = {
        'noisy_mean': 0, 'noisy_std': 1, 'clean_mean': 0, 'clean_std': 1,
        'noisy_min': -1, 'noisy_max': 1, 'clean_min': -1, 'clean_max': 1
    }

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# The data is already normalized in the data preparation step, so we don't need to normalize again

# Define model
model = Sequential([
    tf.keras.layers.LSTM(128, input_shape=(512, 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile model
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
    metrics=['mae']
)
model.summary()

# Output directory
output_dir = './lstm_model_output'
os.makedirs(output_dir, exist_ok=True)

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, 'model.weights.h5'),
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=1e-6
)

class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'Epoch {epoch + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'loss_plot_epoch_{epoch + 1}.png'))
        plt.close()

# Train
batch_size = 128
epochs = 100
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stop, reduce_lr, PlotLosses()],
    verbose=1
)
print("Training completed successfully!")

# Predict
print("Generating predictions...")
y_pred = model.predict(X_test, batch_size=batch_size)
np.save(os.path.join(output_dir, 'predictions.npy'), y_pred)

# Function to denormalize data based on which normalization was used
def denormalize(data, mean, std, min_val, max_val):
    # If we used standardization (mean=0, std=1)
    return data * std + mean
    
    # If we used min-max normalization to [-1, 1]
    # return (data + 1) / 2 * (max_val - min_val) + min_val

# Denormalize data for visualization
X_test_denorm = denormalize(X_test, norm_stats['noisy_mean'], norm_stats['noisy_std'], 
                           norm_stats['noisy_min'], norm_stats['noisy_max'])
y_test_denorm = denormalize(y_test, norm_stats['clean_mean'], norm_stats['clean_std'],
                           norm_stats['clean_min'], norm_stats['clean_max'])
y_pred_denorm = denormalize(y_pred, norm_stats['clean_mean'], norm_stats['clean_std'],
                           norm_stats['clean_min'], norm_stats['clean_max'])

# Plot examples
for i in range(min(10, len(X_test))):
    plt.figure(figsize=(12, 8))
    
    # Plot noisy signal (input)
    plt.subplot(3, 1, 1)
    plt.title(f"Example {i+1}: Noisy Signal (Input)")
    plt.plot(X_test_denorm[i, :, 0])
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot predicted clean signal (output)
    plt.subplot(3, 1, 2)
    plt.title("Predicted Clean Signal (Output)")
    plt.plot(y_pred_denorm[i, :, 0])
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot ground truth clean signal (target)
    plt.subplot(3, 1, 3)
    plt.title("Ground Truth Clean Signal (Target)")
    plt.plot(y_test_denorm[i, :, 0])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'example_{i}.png'))
    plt.close()

# Plot a few examples with all signals on the same plot to compare amplitudes
for i in range(min(5, len(X_test))):
    plt.figure(figsize=(15, 5))
    plt.title(f"Signal Comparison (Example {i+1})")
    plt.plot(X_test_denorm[i, :, 0], label='Noisy (Input)', alpha=0.7)
    plt.plot(y_pred_denorm[i, :, 0], label='Predicted (Output)', alpha=0.7)
    plt.plot(y_test_denorm[i, :, 0], label='Clean (Target)', alpha=0.7)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{i}.png'))
    plt.close()

print("Predictions and example plots saved!")