import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def view_npy_file(filepath):
    """
    Load and display a .npy file.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        # Load the numpy array
        data = np.load(filepath)
        print("--------------------------------------------------")
        print(f"Successfully loaded: {os.path.basename(filepath)}")
        print(f"Data type: {data.dtype}")
        print(f"Data shape: {data.shape}")
        
        # Check if the data is likely an image
        if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in [1, 3, 4]):
            print("Data appears to be an image. Launching visualizer...")
            
            plt.figure(figsize=(8, 6))
            
            # If it's a 3-channel RGB image saved via OpenCV/our script,
            # OpenCV might have saved it as BGR under the hood depending on generation,
            # but since we ensured in our create_dataset to save pure channels, or converted RGBs:
            if data.ndim == 2:
                # Plot grayscale/single channel
                plt.imshow(data, cmap='gray')
            else:
                # If it's the full RGB/HSV matrix (3 channels)
                
                # Check if it was saved as BGR matrix from cv2.imread without standardizing to RGB
                # In create_dataset we saved img_rgb (which is converted to RGB) for full_RGB.npy
                # For HSV/YCbCr, matplotlib will interpret them as RGB so colors will look 'weird', 
                # but it correctly visualizes the raw data.
                plt.imshow(data)
                
            plt.title(f"Viewing: {os.path.basename(filepath)} | Shape: {data.shape}")
            plt.axis('on') # Show axes for pixel coordinates
            print("Close the pop-up window to exit.")
            plt.show()
        else:
            print("Data is not a standard image shape.")
            print("Here is a sample of the data values:")
            print(data)
        print("--------------------------------------------------")
            
    except Exception as e:
        print(f"Failed to read the .npy file. Error: {e}")

if __name__ == "__main__":
    import sys
    
    # Simple interactive setup if run without arguments
    if len(sys.argv) == 1:
        print("=== Numpy (.npy) File Viewer ===")
        filepath_input = input("Please drag and drop a .npy file here (or type the path) and press Enter:\n> ").strip()
        # Remove surrounding quotes if the user dragged and dropped
        filepath_input = filepath_input.strip("\"'")
        view_npy_file(filepath_input)
    else: # CLI Setup
        parser = argparse.ArgumentParser(description="View a .npy data file")
        parser.add_argument("filepath", help="Path to the .npy file")
        args = parser.parse_args()
        view_npy_file(args.filepath)
