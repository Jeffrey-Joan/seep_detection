from utils import extract_instance, paste_instance
import os
import cv2
from tqdm import tqdm
import numpy as np

def instance_augmentation(image_dir, mask_dir, output_image_dir, output_mask_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    plain = cv2.imread(image_dir + "041872.000084.tif" )

    for mask_file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_file)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if len(np.unique(mask_image)) > 2:
            original_image_path = os.path.join(image_dir, mask_file)
            original_image = cv2.imread(original_image_path)
            ids = np.unique(mask_image)

            for instance_id in ids:
                if instance_id != 0:
                    roi, instance_mask = extract_instance(original_image, mask_image, instance_id)
                    
                    # Reset base image for each instance
                    sar_plain = plain.copy()
                    result_image = paste_instance(sar_plain, roi, instance_mask)

                    sar_image_path = os.path.join(output_image_dir, f"{mask_file[:-4]}.{instance_id}.tif")
                    instance_mask_path = os.path.join(output_mask_dir, f"{mask_file[:-4]}.{instance_id}.tif")
                    
                    cv2.imwrite(sar_image_path, result_image)
                    cv2.imwrite(instance_mask_path, instance_mask)

