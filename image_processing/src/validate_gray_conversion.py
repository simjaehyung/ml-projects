import cv2
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, 'assets', 'test_image_original.jpg')
    
    if not os.path.exists(img_path):
        print(f"Error: Could not find {img_path}")
        return

    # Load image safely handling Korean paths if necessary
    with open(img_path, 'rb') as f:
        img_bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

    # 1. OpenCV Method (Weighted Average based on human perception)
    # Y = 0.299 R + 0.587 G + 0.114 B
    gray_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Mean Method (Simple Average)
    # Y = (R + G + B) / 3
    img_float = img_bgr.astype(np.float32)
    gray_mean = np.mean(img_float, axis=2).astype(np.uint8)

    # 3. Calculate absolute difference per pixel
    diff = cv2.absdiff(gray_cv2, gray_mean)

    # Calculate statistics
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    
    print("=== Grayscale Conversion Validation ===")
    print(f"Mean Pixel Difference: {mean_diff:.2f} (out of 255)")
    print(f"Max Pixel Difference:  {max_diff} (out of 255)")
    print(f"Min Pixel Difference:  {min_diff} (out of 255)")
    
    # 4. Enhance the difference for visualization
    # We multiply the difference so the maximum difference becomes 255 (brightest white/red)
    # This makes even subtle differences clearly visible.
    scale_factor = 255.0 / max(max_diff, 1)
    enhanced_diff = np.clip(diff * scale_factor, 0, 255).astype(np.uint8)

    # Apply a Jet colormap (Blue = 0 difference, Green/Yellow = Medium, Red = Max difference)
    heatmap = cv2.applyColorMap(enhanced_diff, cv2.COLORMAP_JET)

    # Save outputs
    diff_path = os.path.join(base_dir, 'assets', 'test_image_diff_enhanced.jpg')
    heatmap_path = os.path.join(base_dir, 'assets', 'test_image_diff_heatmap.jpg')
    
    cv2.imwrite(diff_path, enhanced_diff)
    cv2.imwrite(heatmap_path, heatmap)
    
    print(f"\nVisualizations saved:")
    print(f"1. Enhanced Gray Diff: {diff_path}")
    print(f"2. Color Heatmap Diff: {heatmap_path}")
    print("=======================================")

if __name__ == '__main__':
    main()
