import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import io, measure
from skimage.color import label2rgb
from skimage import io, exposure, img_as_float, img_as_ubyte, filters, img_as_float

from preprocess import lee_filter

from utils import mask_to_polygons_multi, annotations_to_mask, overlay_mask_on_image_with_boxes

colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
        (128, 128, 128) # Gray
    ]

def plot_image_and_mask_path(image_path, mask_path):
    # Read the image and mask
    image = io.imread(image_path)
    mask = io.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Create a color overlay of the mask on the image
    mask_overlay = label2rgb(mask, image=image, bg_label=0, alpha=0.3)

    # Plot the image with mask overlay
    ax2.imshow(mask_overlay)
    ax2.set_title('Image with Mask Overlay')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def plot_image_and_mask(image, mask):
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Create a color overlay of the mask on the image
    mask_overlay = label2rgb(mask, image=image, bg_label=0, alpha=0.3)

    # Plot the image with mask overlay
    ax2.imshow(mask_overlay)
    ax2.set_title('Image with Mask Overlay')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def process_and_plot_image(image, clip_limit=0.03, sigma=1, size=256):
    
    # Normalize the image to [0, 1] for consistency in processing
    image_float = img_as_float(image)
    
    # Apply Histogram Equalization
    hist_eq = exposure.equalize_hist(image_float)
    
    # Apply Adaptive Histogram Equalization
    adapt_hist_eq = exposure.equalize_adapthist(image_float, clip_limit=clip_limit)
    
    # Apply Gaussian Filter
    gaussian_filtered = filters.gaussian(image_float, sigma=sigma)
    
    # Apply Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter((image_float * 255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
    
    # Invert Image
    inverted = 1 - image_float
    
    # Normalize the image
    normalized_image = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float))
    
    # Apply Lee Filter (Spectral Filter)
    lee_filtered = lee_filter(image_float, size=size)
    
    # Plot the original and processed images using subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # Titles for the subplots
    titles = ['Original Image', 'Histogram Equalization', 'Adaptive Histogram Equalization', 
              'Gaussian Filter', 'Bilateral Filter', 'Inverted Image', 'Normalized Image', 'Spectral Filter (Lee)']
    
    # Normalize the original image for consistency in plotting
    original_image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # All images to plot
    images = [original_image_normalized, hist_eq, adapt_hist_eq, gaussian_filtered, 
              bilateral_filtered, inverted, normalized_image, lee_filtered]
    
    # Plot all images
    for i in range(8):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Show the figure
    plt.tight_layout()
    plt.show()

def hist_lee_plot_image(image, size=256):
    
    # Normalize the image to [0, 1] for consistency in processing
    image_float = img_as_float(image)
    
    # Apply Histogram Equalization
    hist_eq = exposure.equalize_hist(image_float)
    
    # Apply Lee Filter after histogram equalization
    lee_filtered = lee_filter(hist_eq, size=size)
    
    # Plot the original and processed images using subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Titles for the subplots
    titles = ['Original Image', 'Histogram Equalization + Lee Filter']
    
    # Normalize the original image for consistency in plotting
    original_image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # All images to plot
    images = [original_image_normalized, lee_filtered]
    
    # Plot all images
    for i in range(2):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    

    plt.tight_layout()
    plt.show()

def plot_lee_filters(image):
    
    # Normalize the image to [0, 1] for consistency in processing
    image_float = img_as_float(image)
    

    
    lee_4 = lee_filter(image_float,4)

    lee_8 = lee_filter(image_float,8)
    
    lee_16 = lee_filter(image_float,16)

    lee_32 = lee_filter(image_float,32)

    lee_64 = lee_filter(image_float,64)

    lee_128 = lee_filter(image_float,128)

    lee_256 = lee_filter(image_float,256)
    # Plot the original and processed images using subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # Titles for the subplots
    titles = ['Original Image', 'Lee Filter size 4', 'Lee Filter size 8', 
              'Lee Filter size 16', 'Lee Filter size 32', 'Lee Filter size 64', 'Lee Filter size 128', 'Lee Filter size 256']
    
    # Normalize the original image for consistency in plotting
    original_image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # All images to plot
    images = [original_image_normalized, lee_4, lee_8, lee_16, 
              lee_32, lee_64, lee_128, lee_256]
    
    # Plot all images
    for i in range(8):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Show the figure
    plt.tight_layout()
    plt.show()


def plot_poly_overlay_with_bounding_box(image, mask, image_shape=(256,256)):
    
    polygons, labels, annotations = mask_to_polygons_multi(mask)
    mask = annotations_to_mask(annotations, image_shape[:2])
    overlay = overlay_mask_on_image_with_boxes(image, mask, annotations)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(overlay)
    ax2.set_title("Polygon Overlay with Bounding Boxes")
    ax2.axis('off')

    # Add legend for classes with correct colors and category IDs
    unique_categories = set(ann['category_id'] for ann in annotations)
    handles = [plt.Line2D([0], [0], color=np.array(colors[cat_id % len(colors)]) / 255.0, lw=4) for cat_id in unique_categories]
    labels = [f"Label {cat_id}" for cat_id in unique_categories]
    fig2.legend(handles=handles, labels=labels)

    plt.tight_layout()
    plt.show()


def plot_image_label_distribution(mask_dir):
    label_counts = {label: 0 for label in range(1, 8)}  # Initialize counts for labels 1 to 7

    for mask_file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_file)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_labels = np.unique(mask_image)

        for label in unique_labels:
            if label != 0:  # Skip zeros (background)
                label_counts[label] += 1

    # Plotting the histogram
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Labels in Images (Excluding Background)')
    plt.xticks(labels)
    plt.show()


def plot_images_and_masks(images, rois):
    fig, axs = plt.subplots(2, len(images), figsize=(15, 10))
    
    # Plot images in the first row
    for i in range(len(images)):
        axs[0, i].imshow(images[i], cmap='gray')
        axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].axis('off')
    
    # Plot ROIs in the second row
    for i in range(len(images)):
        axs[1, i].imshow(rois[i], cmap='gray')
        axs[1, i].set_title(f'Mask {i+1}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_a_plot(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 16)) 
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_predictions_labels(parent_dir):

    image_files = [
        'val_batch0_labels.jpg', 'val_batch1_labels.jpg', 
        'val_batch2_labels.jpg','val_batch0_pred.jpg', 
        'val_batch1_pred.jpg', 'val_batch2_pred.jpg'
    ]
    
    # Create a 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Prediction and Labels', fontsize=16)
    

    axes = axes.flatten()
    
    for i, filename in enumerate(image_files):
        img_path = os.path.join(parent_dir, filename)
        if os.path.isfile(img_path):

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img)
            #axes[i].set_title(filename)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Image not found:\n{filename}", 
                         ha='center', va='center', wrap=True)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_test_curves(parent_dir):
    image_files = [
        'MaskPR_curve.png', 'MaskR_curve.png', 'BoxR_curve.png', 'MaskF1_curve.png',
        'MaskP_curve.png', 'BoxF1_curve.png', 'BoxP_curve.png', 'BoxPR_curve.png'
    ]
    
    n_images = len(image_files)
    n_cols = 3  
    n_rows = math.ceil(n_images / n_cols)
    

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    fig.suptitle('Test Curves and Validation Images', fontsize=16)
    
    axes = axes.flatten()
    
    for i, filename in enumerate(image_files):
        img_path = os.path.join(parent_dir, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Plot the image
            axes[i].imshow(img)
            axes[i].set_title(filename, fontsize=10)
            axes[i].axis('off')
    for i in range(n_images, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()