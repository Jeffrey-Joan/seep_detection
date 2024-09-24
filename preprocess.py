from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import cv2
import os
from skimage import io, exposure
import numpy as np


def lee_filter(img, size):
    if img.ndim == 3:
        return np.stack([lee_filter(img[..., i], size) for i in range(img.shape[-1])], axis=-1)

    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def preprocess_hist_lee(image_path, size=256):
    
    # Normalize the image to [0, 1] for consistency in processing
    image = cv2.imread(image_path)
    
    # Apply Histogram Equalization
    hist_eq = exposure.equalize_hist(image)
    
    # Apply Lee Filter after histogram equalization
    lee_filtered = lee_filter(hist_eq, size=size)
    
    return lee_filtered

def process_dir_save(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            image_path = os.path.join(input_dir, filename)
            
            processed_image = preprocess_hist_lee(image_path)

            # Normalize to [0, 1] range
            processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min())
            
            # Convert to uint8 [0, 255] range
            processed_image = (processed_image * 255).astype(np.uint8)

            output_path = os.path.join(output_dir, filename.replace('.tif', '.png'))
            cv2.imwrite(output_path, processed_image)

    print(f"Processed {len(os.listdir(input_dir))} images")

def convert_labels_to_single_seep(input_directory, output_directory):

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # List all TXT files in the input directory
        label_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
        
        for label_file in label_files:
            input_file_path = os.path.join(input_directory, label_file)
            output_file_path = os.path.join(output_directory, label_file)
            
            with open(input_file_path, 'r') as input_file:
                lines = input_file.readlines()
                
            # Convert all lines to start with '0' (seep class) while maintaining polygon coordinates
            modified_lines = [f"0 {' '.join(line.split()[1:])}\n" for line in lines]
            
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(modified_lines)
