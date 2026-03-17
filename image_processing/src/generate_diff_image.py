import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'jean12.jpg'
try:
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Extract channels
R = img_rgb[:,:,0].astype(np.float32)
G = img_rgb[:,:,1].astype(np.float32)
B = img_rgb[:,:,2].astype(np.float32)

# Method A: Average
gray_A = (R + G + B) / 3.0

# Method B: Luminosity (ITU-R BT.601)
gray_B = 0.299 * R + 0.587 * G + 0.114 * B

# Calculate the difference (Absolute difference to see magnitude of change)
difference = np.abs(gray_A - gray_B)

# Create a figure to save
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Method A: Average (R+G+B)/3')
plt.imshow(gray_A, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Method B: Luminosity (Weighted)')
plt.imshow(gray_B, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference Map (Heatmap)')
# Use a heatmap to clearly show where the differences are largest
plt.imshow(difference, cmap='hot')
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

plt.tight_layout()
output_path = 'grayscale_difference_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully saved difference analysis image to {output_path}")
