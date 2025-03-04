import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential  # Correct import for Sequential

# Force eager execution
tf.config.run_functions_eagerly(True)

# Explicitly disable TPU usage and stick with CPU
try:
    tf.config.set_visible_devices([], 'TPU')
    print("TPU devices hidden")
except:
    print("No TPU devices found to hide")

print("TensorFlow version:", tf.__version__)
print("Eager execution enabled:", tf.executing_eagerly())

# Load your data
data_dir = './MIT-BIH-ST-Dataset/Train_Test_Data'
X_train = np.load(os.path.join(data_dir, 'mit_st_X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'mit_st_y_train.npy'))
X_test = np.load(os.path.join(data_dir, 'mit_st_X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'mit_st_y_test.npy'))

# Print data shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Ensure data is in the right format
if len(X_train.shape) == 2:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

# Define the new model
model = Sequential()
model.add(tf.keras.layers.LSTM(140, input_shape=(512, 1), return_sequences=True))
model.add(tf.keras.layers.Dense(140, activation='relu'))
model.add(tf.keras.layers.LSTM(140, return_sequences=True))  # Second LSTM layer
model.add(tf.keras.layers.Dense(140, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.summary()

# Create output directory
output_dir = './lstm_model_output'
os.makedirs(output_dir, exist_ok=True)

# Compile model with standard Adam optimizer
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)  # Standard Adam
)

# Minimal callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, 'model.weights.h5'),
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

# Training parameters
batch_size = 64
epochs = 5

try:
    # Train the model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    print("Training completed successfully!")
    
    # Save predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test, batch_size=batch_size)
    np.save(os.path.join(output_dir, 'predictions.npy'), y_pred)
    print("Predictions saved!")
    
except Exception as e:
    print(f"Error during execution: {str(e)}")
    
    # Fallback to a simpler approach
    print("\n\nTrying an alternative approach...")
    
    # Create a simple dense model
    simple_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(512, 1)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(512)
    ])
    
    simple_model.compile(optimizer='adam', loss='mse')
    
    # Reshape targets for dense model
    y_train_flat = y_train.reshape(y_train.shape[0], -1)  # Flatten the targets
    
    print("Training simple model...")
    simple_model.fit(
        X_train, y_train_flat,
        batch_size=16,
        epochs=5,
        verbose=1
    )
    
    print("Simple model training complete!")