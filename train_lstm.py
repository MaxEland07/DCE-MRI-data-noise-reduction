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

# Load normalization statistics
norm_stats_path = os.path.join(data_dir, 'normalization_stats.npy')
if os.path.exists(norm_stats_path):
    print("Loading pre-computed normalization statistics...")
    norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
    norm_stats = {'mean': norm_stats['mean'], 'std': norm_stats['std']}
else:
    print("No normalization statistics found. Data assumed normalized.")
    norm_stats = {'mean': 0, 'std': 1}

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Define model from IEEE paper with added dropout
def improved_ecg_denoising_model():
    """
    Hybrid CNN-LSTM-Attention model for ECG denoising
    Combines benefits of CNN for local feature extraction, LSTM for temporal relationships,
    and attention mechanism for focusing on important signal aspects
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(512, 1)),
        
        # CNN layers for local feature extraction
        tf.keras.layers.Conv1D(64, kernel_size=9, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # Bidirectional LSTM for temporal dependencies (forward and backward)
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        
        # Second Bidirectional LSTM layer
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        
        # Attention mechanism to focus on relevant parts of the signal
        tf.keras.layers.Attention()([
            tf.keras.layers.Dense(128)(tf.keras.layers.LayerNormalization()), 
            tf.keras.layers.Dense(128)(tf.keras.layers.LayerNormalization())
        ]),
        
        # Final output layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    return model

model = improved_ecg_denoising_model()

# Custom SNR metric (fixed to use tf.math.log)
def snr_metric(y_true, y_pred):
    signal_power = tf.reduce_mean(tf.square(y_true))
    noise_power = tf.reduce_mean(tf.square(y_true - y_pred))
    # Use tf.math.log and change of base: log10(x) = log(x) / log(10)
    log10_signal_noise = tf.math.log(signal_power / (noise_power + 1e-10)) / tf.math.log(10.0)
    return 10.0 * log10_signal_noise

# Compile model
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    metrics=['mae', snr_metric]
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
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
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
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'Epoch {epoch + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'loss_plot_epoch_{epoch + 1}.png'))
        plt.close()

# Train
batch_size = 64
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

# Denormalize data
def denormalize(data, mean, std):
    return data * std + mean

X_test_denorm = denormalize(X_test, norm_stats['mean'], norm_stats['std'])
y_test_denorm = denormalize(y_test, norm_stats['mean'], norm_stats['std'])
y_pred_denorm = denormalize(y_pred, norm_stats['mean'], norm_stats['std'])

# Plot examples
for i in range(min(len(X_test), 10)):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title(f"Example {i+1}: Noisy Signal (Input)")
    plt.plot(X_test_denorm[i, :, 0])
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.title("Predicted Clean Signal (Output)")
    plt.plot(y_pred_denorm[i, :, 0])
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.title("Ground Truth Clean Signal (Target)")
    plt.plot(y_test_denorm[i, :, 0])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'example_{i}.png'))
    plt.close()

# Plot comparisons
for i in range(min(len(X_test), 5)):
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