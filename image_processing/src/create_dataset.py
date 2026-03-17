import cv2
import numpy as np
import os
import shutil

def imwrite_korean(filename, img):
    extension = os.path.splitext(filename)[1]
    result, n_array = cv2.imencode(extension, img)
    if result:
        with open(filename, mode='wb') as f:
            n_array.tofile(f)

def process_directory(base_dir, output_dir):
    # Process specific images found in Guidecode and assign a numerical index
    images_to_process = {
        1: 'dog_color.png', 
        2: 'hanyang.jpg', 
        3: 'hanyang_mask.jpg', 
        4: 'wall.PNG'
    }
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create the index mapping text file
    with open(os.path.join(output_dir, 'index_mapping.txt'), 'w', encoding='utf-8') as f:
        f.write("Dataset Index Mapping:\n")
        f.write("-" * 25 + "\n")
        for idx, name in images_to_process.items():
            f.write(f"Index {idx} -> {name}\n")
            
    for img_idx, img_name in images_to_process.items():
        img_path = os.path.join(base_dir, img_name)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
            
        # Create a directory for this specific image's dataset using the numerical index
        img_out_dir = os.path.join(output_dir, str(img_idx))
        
        # Load the image (Using np.fromfile to handle Korean paths)
        img_array = np.fromfile(img_path, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        print(f"Processing: {img_name}...")

        # ---------------------------
        # 1. RGB Domain
        # ---------------------------
        rgb_dir = os.path.join(img_out_dir, 'RGB')
        os.makedirs(rgb_dir, exist_ok=True)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Save full RGB image
        imwrite_korean(os.path.join(rgb_dir, 'full_RGB.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)) # convert back to BGR for imwrite to save correctly as RGB looking
        np.save(os.path.join(rgb_dir, 'full_RGB.npy'), img_rgb) # Save raw array
        
        # Split RGB channels
        R, G, B = cv2.split(img_rgb)
        
        # To visualize individual channels as color images, we merge them with zeros
        zeros = np.zeros_like(R)
        img_R = cv2.merge([zeros, zeros, R]) # OpenCV uses BGR for saving
        img_G = cv2.merge([zeros, G, zeros])
        img_B = cv2.merge([B, zeros, zeros])
        
        imwrite_korean(os.path.join(rgb_dir, 'R.jpg'), img_R)
        imwrite_korean(os.path.join(rgb_dir, 'G.jpg'), img_G)
        imwrite_korean(os.path.join(rgb_dir, 'B.jpg'), img_B)
        
        # Save raw numeric channel arrays
        np.save(os.path.join(rgb_dir, 'R.npy'), R)
        np.save(os.path.join(rgb_dir, 'G.npy'), G)
        np.save(os.path.join(rgb_dir, 'B.npy'), B)

        # ---------------------------
        # 2. HSV Domain
        # ---------------------------
        hsv_dir = os.path.join(img_out_dir, 'HSV')
        os.makedirs(hsv_dir, exist_ok=True)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Save full HSV image
        imwrite_korean(os.path.join(hsv_dir, 'full_HSV.jpg'), img_hsv)
        np.save(os.path.join(hsv_dir, 'full_HSV.npy'), img_hsv)
        
        # Split HSV channels (Hue, Saturation, Value)
        H, S, V = cv2.split(img_hsv)
        
        # Save them as grayscale images to see the intensity of each component
        imwrite_korean(os.path.join(hsv_dir, 'Hue.jpg'), H)
        imwrite_korean(os.path.join(hsv_dir, 'Saturation.jpg'), S)
        imwrite_korean(os.path.join(hsv_dir, 'Value.jpg'), V)
        
        # Save raw numeric channel arrays
        np.save(os.path.join(hsv_dir, 'H.npy'), H)
        np.save(os.path.join(hsv_dir, 'S.npy'), S)
        np.save(os.path.join(hsv_dir, 'V.npy'), V)

        # ---------------------------
        # 3. YCbCr Domain
        # ---------------------------
        ycbcr_dir = os.path.join(img_out_dir, 'YCbCr')
        os.makedirs(ycbcr_dir, exist_ok=True)
        img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb) # Note: OpenCV uses YCrCb order
        
        # Save full YCbCr
        imwrite_korean(os.path.join(ycbcr_dir, 'full_YCbCr.jpg'), img_ycbcr)
        np.save(os.path.join(ycbcr_dir, 'full_YCbCr.npy'), img_ycbcr)
        
        # Split channels (Y, Cr, Cb)
        Y, Cr, Cb = cv2.split(img_ycbcr)
        
        # Save them as grayscale images
        imwrite_korean(os.path.join(ycbcr_dir, 'Y_Luma.jpg'), Y)
        imwrite_korean(os.path.join(ycbcr_dir, 'Cr_Chroma_Red.jpg'), Cr)
        imwrite_korean(os.path.join(ycbcr_dir, 'Cb_Chroma_Blue.jpg'), Cb)
        
        # Save raw numeric channel arrays
        np.save(os.path.join(ycbcr_dir, 'Y.npy'), Y)
        np.save(os.path.join(ycbcr_dir, 'Cr.npy'), Cr)
        np.save(os.path.join(ycbcr_dir, 'Cb.npy'), Cb)

    print("\nProcessing complete! Dataset has been generated in:", output_dir)
    
    # Zip the dataset
    zip_path = output_dir # shutil.make_archive adds the .zip extension automatically
    shutil.make_archive(zip_path, 'zip', output_dir)
    print(f"Zipped dataset saved to: {zip_path}.zip")

class DatasetRetriever:
    """Helper class to load dataset images using only a dummy numerical index."""
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.domain_map = {
            'RGB': {'full': 'full_RGB', 'R': 'R', 'G': 'G', 'B': 'B'},
            'HSV': {'full': 'full_HSV', 'H': 'H', 'S': 'S', 'V': 'V'},
            'YCbCr': {'full': 'full_YCbCr', 'Y': 'Y', 'Cb': 'Cb', 'Cr': 'Cr'}
        }
        
    def get_image(self, index, domain='RGB', channel='full', load_raw_npy=True):
        """
        Retrieves an image array by its numerical index.
        Args:
            index (int): The image ID (1=dog_color, 2=hanyang, 3=hanyang_mask, 4=wall)
            domain (str): 'RGB', 'HSV', or 'YCbCr'
            channel (str): 'full', or specific channels ('R','G','B', 'H','S','V', 'Y','Cb','Cr')
            load_raw_npy (bool): If True, loads the pure numerical data array (.npy). If False, loads the visualization .jpg image.
        Returns:
            numpy array of the image/data, or None if not found.
        """
        if domain not in self.domain_map or channel not in self.domain_map[domain]:
            print(f"Invalid domain ({domain}) or channel ({channel}).")
            return None
            
        base_filename = self.domain_map[domain][channel]
        
        if load_raw_npy:
            file_path = os.path.join(self.dataset_dir, str(index), domain, base_filename + '.npy')
            if not os.path.exists(file_path):
                print(f"Raw data file not found: {file_path}")
                return None
            return np.load(file_path)
        else:
            # Revert to the mapped jpg name for visualization loading
            # (Reconstructing the jpg filename correctly since we streamlined the map for .npy)
            jpg_map = {
                'RGB': {'full': 'full_RGB.jpg', 'R': 'R.jpg', 'G': 'G.jpg', 'B': 'B.jpg'},
                'HSV': {'full': 'full_HSV.jpg', 'H': 'Hue.jpg', 'S': 'Saturation.jpg', 'V': 'Value.jpg'},
                'YCbCr': {'full': 'full_YCbCr.jpg', 'Y': 'Y_Luma.jpg', 'Cb': 'Cb_Chroma_Blue.jpg', 'Cr': 'Cr_Chroma_Red.jpg'}
            }
            file_path = os.path.join(self.dataset_dir, str(index), domain, jpg_map[domain][channel])
            if not os.path.exists(file_path):
                print(f"Image visualization file not found: {file_path}")
                return None
            img_array = np.fromfile(file_path, np.uint8)
            read_flag = cv2.IMREAD_COLOR if channel == 'full' or domain == 'RGB' else cv2.IMREAD_GRAYSCALE
            return cv2.imdecode(img_array, read_flag)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_directory = os.path.join(project_root, 'notebooks')
    output_directory = os.path.join(project_root, 'data', 'raw')

    # 1. Clean up old text-named folders if they exist
    old_folders = ['dog_color', 'hanyang', 'hanyang_mask', 'wall']
    for old_f in old_folders:
        old_path = os.path.join(output_directory, old_f)
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
    
    # 2. Run processing to generate number-indexed dataset
    process_directory(base_directory, output_directory)
    
    # 3. Example of how to easily fetch by numbers!
    print("\n--- Testing Retrieval Helper ---")
    retriever = DatasetRetriever(output_directory)
    
    # Fetch image ID 2 in HSV domain (Raw Numpy Data)
    test_num_data = retriever.get_image(index=2, domain='HSV', channel='full', load_raw_npy=True)
    if test_num_data is not None:
        print(f"Successfully loaded Raw Numpy Dataset for ID 2 (HSV-full). Shape: {test_num_data.shape}, dtype: {test_num_data.dtype}")
        
    # Fetch image ID 4 in RGB domain, Red channel (Raw Numpy Data)
    test_num_data2 = retriever.get_image(index=4, domain='RGB', channel='R', load_raw_npy=True)
    if test_num_data2 is not None:
        print(f"Successfully loaded Raw Numpy Dataset for ID 4 (RGB-Red). Shape: {test_num_data2.shape}, dtype: {test_num_data2.dtype}")
