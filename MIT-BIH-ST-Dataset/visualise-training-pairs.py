import sys
# Add paths to your installed packages - replace with your actual paths
sys.path.append("C:/Users/maxel/AppData/Local/Programs/Python/Python39/Lib/site-packages")
# Now import the packages
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory where your processed data is stored
base_dir = './MIT-BIH-ST-Dataset'
processed_data_dir = os.path.join(base_dir, 'Train_Test_Data')

# Function to load and visualize training pairs
def visualize_training_pairs(num_examples=5):
    """
    Load and visualize pairs of noisy and clean signals from the training data
    
    Args:
        num_examples: Number of examples to visualize (default: 5)
    """
    try:
        # Load the training data
        X_train = np.load(os.path.join(processed_data_dir, 'mit_st_X_train.npy'))
        y_train = np.load(os.path.join(processed_data_dir, 'mit_st_y_train.npy'))
        
        print(f"Data loaded successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        # Visualize the specified number of examples
        for i in range(min(num_examples, len(X_train))):
            plt.figure(figsize=(12, 6))
            
            # Plot the noisy signal (input)
            plt.subplot(2, 1, 1)
            plt.plot(X_train[i].flatten())
            plt.title(f'Example {i+1}: Noisy Signal (Input)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Plot the clean signal (target)
            plt.subplot(2, 1, 2)
            plt.plot(y_train[i].flatten())
            plt.title(f'Example {i+1}: Clean Signal (Target)')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run the Download-MIT-Kaggle.py script first to generate the training data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    visualize_training_pairs()
