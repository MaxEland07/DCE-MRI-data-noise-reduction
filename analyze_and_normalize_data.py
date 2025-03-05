import numpy as np
import os
import matplotlib.pyplot as plt

# Define data directories
data_dir = './MIT-BIH-ST-Dataset/Train_Test_Data'
output_dir = './MIT-BIH-ST-Dataset/Normalized_Data'
os.makedirs(output_dir, exist_ok=True)

# Load existing data
print("Loading existing data...")
X_train = np.load(os.path.join(data_dir, 'mit_st_X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'mit_st_y_train.npy'))
X_test = np.load(os.path.join(data_dir, 'mit_st_X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'mit_st_y_test.npy'))

print(f"Data shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# Analyze amplitude ranges
print("\nCurrent amplitude ranges:")
print(f"X_train (noisy): min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}, std={X_train.std():.4f}")
print(f"y_train (clean): min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"X_test (noisy): min={X_test.min():.4f}, max={X_test.max():.4f}, mean={X_test.mean():.4f}, std={X_test.std():.4f}")
print(f"y_test (clean): min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}, std={y_test.std():.4f}")

# Calculate global statistics
all_noisy = np.vstack((X_train, X_test))
all_clean = np.vstack((y_train, y_test))

normalization_stats = {
    'noisy_mean': float(np.mean(all_noisy)),
    'noisy_std': float(np.std(all_noisy)),
    'clean_mean': float(np.mean(all_clean)),
    'clean_std': float(np.std(all_clean)),
    'noisy_min': float(np.min(all_noisy)),
    'noisy_max': float(np.max(all_noisy)),
    'clean_min': float(np.min(all_clean)),
    'clean_max': float(np.max(all_clean))
}

print("\nGlobal statistics:")
for key, value in normalization_stats.items():
    print(f"{key}: {value:.4f}")

# Plot examples before normalization
plots_dir = './MIT-BIH-ST-Dataset/Analysis_Plots'
os.makedirs(plots_dir, exist_ok=True)

for i in range(min(5, len(X_test))):
    plt.figure(figsize=(15, 10))
    
    # Before normalization
    plt.subplot(2, 1, 1)
    plt.title(f"Before Normalization (Example {i+1})")
    plt.plot(X_test[i, :, 0], label='Noisy Input', alpha=0.7)
    plt.plot(y_test[i, :, 0], label='Clean Target', alpha=0.7)
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    # Apply normalization to this example
    
    # Option 1: Standardization (z-score normalization)
    X_norm = (X_test[i, :, 0] - normalization_stats['noisy_mean']) / normalization_stats['noisy_std']
    y_norm = (y_test[i, :, 0] - normalization_stats['clean_mean']) / normalization_stats['clean_std']
    
    # Option 2: Min-max normalization to [-1, 1] range
    # X_norm = 2 * (X_test[i, :, 0] - normalization_stats['noisy_min']) / (normalization_stats['noisy_max'] - normalization_stats['noisy_min']) - 1
    # y_norm = 2 * (y_test[i, :, 0] - normalization_stats['clean_min']) / (normalization_stats['clean_max'] - normalization_stats['clean_min']) - 1
    
    plt.subplot(2, 1, 2)
    plt.title("After Normalization")
    plt.plot(X_norm, label='Normalized Noisy Input', alpha=0.7)
    plt.plot(y_norm, label='Normalized Clean Target', alpha=0.7)
    plt.xlabel("Sample")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'normalization_comparison_{i}.png'))
    plt.close()

print(f"\nExample plots saved to {plots_dir}")

# Apply normalization to all data
print("\nApplying normalization to all data...")

# Option 1: Standardization (z-score normalization)
X_train_norm = (X_train - normalization_stats['noisy_mean']) / normalization_stats['noisy_std']
y_train_norm = (y_train - normalization_stats['clean_mean']) / normalization_stats['clean_std']
X_test_norm = (X_test - normalization_stats['noisy_mean']) / normalization_stats['noisy_std']
y_test_norm = (y_test - normalization_stats['clean_mean']) / normalization_stats['clean_std']

# Option 2: Min-max normalization to [-1, 1] range
# X_train_norm = 2 * (X_train - normalization_stats['noisy_min']) / (normalization_stats['noisy_max'] - normalization_stats['noisy_min']) - 1
# y_train_norm = 2 * (y_train - normalization_stats['clean_min']) / (normalization_stats['clean_max'] - normalization_stats['clean_min']) - 1
# X_test_norm = 2 * (X_test - normalization_stats['noisy_min']) / (normalization_stats['noisy_max'] - normalization_stats['noisy_min']) - 1
# y_test_norm = 2 * (y_test - normalization_stats['clean_min']) / (normalization_stats['clean_max'] - normalization_stats['clean_min']) - 1

# Verify normalized data
print("\nNormalized amplitude ranges:")
print(f"X_train_norm: min={X_train_norm.min():.4f}, max={X_train_norm.max():.4f}, mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
print(f"y_train_norm: min={y_train_norm.min():.4f}, max={y_train_norm.max():.4f}, mean={y_train_norm.mean():.4f}, std={y_train_norm.std():.4f}")
print(f"X_test_norm: min={X_test_norm.min():.4f}, max={X_test_norm.max():.4f}, mean={X_test_norm.mean():.4f}, std={X_test_norm.std():.4f}")
print(f"y_test_norm: min={y_test_norm.min():.4f}, max={y_test_norm.max():.4f}, mean={y_test_norm.mean():.4f}, std={y_test_norm.std():.4f}")

# Save normalized data and statistics
print("\nSaving normalized data...")
np.save(os.path.join(output_dir, 'mit_st_X_train_norm.npy'), X_train_norm)
np.save(os.path.join(output_dir, 'mit_st_y_train_norm.npy'), y_train_norm)
np.save(os.path.join(output_dir, 'mit_st_X_test_norm.npy'), X_test_norm)
np.save(os.path.join(output_dir, 'mit_st_y_test_norm.npy'), y_test_norm)
np.save(os.path.join(output_dir, 'normalization_stats.npy'), normalization_stats)

print("\nNormalization complete!")
print(f"Normalized data saved to {output_dir}")
print("You can now use these normalized files for training your model.")
print("To denormalize predictions, use the normalization_stats.npy file.") 