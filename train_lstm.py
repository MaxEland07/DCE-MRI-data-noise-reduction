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

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Save normalization parameters
mean_X_train, range_X_train = X_train.mean(), X_train.max() - X_train.min()
mean_y_train, range_y_train = y_train.mean(), y_train.max() - y_train.min()
mean_X_test, range_X_test = X_test.mean(), X_test.max() - X_test.min()
mean_y_test, range_y_test = y_test.mean(), y_test.max() - y_test.min()

# Normalize data
X_train = (X_train - mean_X_train) / range_X_train
y_train = (y_train - mean_y_train) / range_y_train
X_test = (X_test - mean_X_test) / range_X_test
y_test = (y_test - mean_y_test) / range_y_test

# Optional: Per-sample standardization (uncomment to use)
# X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / X_train.std(axis=1, keepdims=True)
# y_train = (y_train - y_train.mean(axis=1, keepdims=True)) / y_train.std(axis=1, keepdims=True)
# X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / X_test.std(axis=1, keepdims=True)
# y_test = (y_test - y_test.mean(axis=1, keepdims=True)) / y_test.std(axis=1, keepdims=True)

# Define model
model = Sequential([
    tf.keras.layers.LSTM(128, input_shape=(512, 1), return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
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

# Denormalize
y_pred = y_pred * range_y_test + mean_y_test
X_test_denorm = X_test * range_X_test + mean_X_test
y_test_denorm = y_test * range_y_test + mean_y_test

# Plot examples
for i in range(min(10, len(X_test))):
    plt.figure(figsize=(12, 4))
    plt.subplot(3, 1, 1)
    plt.plot(X_test_denorm[i, :, 0], label='Noisy')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(y_pred[i, :, 0], label='Predicted')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(y_test_denorm[i, :, 0], label='Clean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'example_{i}.png'))
    plt.close()

print("Predictions and example plots saved!")