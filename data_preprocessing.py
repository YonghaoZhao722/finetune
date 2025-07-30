#!/usr/bin/env python3
"""
Data preprocessing script for SAM fine-tuning.
Converts DIC and DIC_mask TIFF files to the format required for training.
"""

import os
import argparse
import shutil
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from scipy import ndimage
from skimage.measure import label
from sklearn.model_selection import train_test_split


def clean_filename(filename):
    """Remove macOS system file prefixes and normalize extensions."""
    # Remove ._* system files prefix
    if filename.startswith('._'):
        return None
    # Remove .DS_Store files
    if filename in ['.DS_Store', '._.DS_Store']:
        return None
    return filename


def get_file_pairs(dic_folder, mask_folder):
    """Get matching pairs of DIC images and masks."""
    dic_files = []
    mask_files = []
    
    # Get clean file lists
    for filename in os.listdir(dic_folder):
        clean_name = clean_filename(filename)
        if clean_name and clean_name.lower().endswith(('.tif', '.tiff')):
            dic_files.append(clean_name)
    
    for filename in os.listdir(mask_folder):
        clean_name = clean_filename(filename)
        if clean_name and clean_name.lower().endswith(('.tif', '.tiff')):
            mask_files.append(clean_name)
    
    # Find matching pairs
    dic_set = set(dic_files)
    mask_set = set(mask_files)
    common_files = dic_set.intersection(mask_set)
    
    print(f"Found {len(dic_files)} DIC images")
    print(f"Found {len(mask_files)} mask images")
    print(f"Found {len(common_files)} matching pairs")
    
    return sorted(list(common_files))


def process_mask(mask_path):
    """
    Process mask to ensure proper instance segmentation format.
    Each object should have a unique ID, background should be 0.
    """
    mask = imread(mask_path)
    
    # Convert to binary if needed
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # If mask is binary, create instance labels
    if len(np.unique(mask)) <= 2:
        # Binary mask - create connected components
        binary_mask = mask > 0
        labeled_mask = label(binary_mask)
    else:
        # Already has instance labels
        labeled_mask = mask
    
    # Ensure background is 0 and labels are consecutive
    unique_labels = np.unique(labeled_mask)
    if 0 not in unique_labels:
        # Shift labels so background becomes 0
        labeled_mask = labeled_mask - unique_labels.min()
    
    # Make labels consecutive
    unique_labels = np.unique(labeled_mask)
    for i, old_label in enumerate(unique_labels):
        if old_label != i:
            labeled_mask[labeled_mask == old_label] = i
    
    return labeled_mask.astype(np.uint16)


def normalize_16bit_to_8bit(image, percentile_normalization=True, lower_percentile=1, upper_percentile=99):
    """
    Normalize 16-bit image to 8-bit range [0, 255].
    
    Args:
        image: Input image array
        percentile_normalization: If True, use percentile-based normalization for better contrast
        lower_percentile: Lower percentile for clipping (default: 1)
        upper_percentile: Upper percentile for clipping (default: 99)
    """
    if image.dtype == np.uint8:
        return image  # Already 8-bit
    
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    if percentile_normalization:
        # Use percentile-based normalization for better contrast
        lower_bound = np.percentile(image_float, lower_percentile)
        upper_bound = np.percentile(image_float, upper_percentile)
        
        # Clip values to percentile range
        image_float = np.clip(image_float, lower_bound, upper_bound)
        
        # Normalize to [0, 1]
        if upper_bound > lower_bound:
            image_float = (image_float - lower_bound) / (upper_bound - lower_bound)
        else:
            image_float = np.zeros_like(image_float)
    else:
        # Simple min-max normalization
        min_val = image_float.min()
        max_val = image_float.max()
        
        if max_val > min_val:
            image_float = (image_float - min_val) / (max_val - min_val)
        else:
            image_float = np.zeros_like(image_float)
    
    # Scale to [0, 255] and convert to uint8
    image_8bit = (image_float * 255).astype(np.uint8)
    
    return image_8bit


def copy_and_process_data(file_pairs, dic_folder, mask_folder, output_folder, split_name, 
                         percentile_norm=True, lower_perc=1, upper_perc=99):
    """Copy and process data files to output folder."""
    output_dic = os.path.join(output_folder, split_name, 'images')
    output_mask = os.path.join(output_folder, split_name, 'masks')
    
    os.makedirs(output_dic, exist_ok=True)
    os.makedirs(output_mask, exist_ok=True)
    
    processed_count = 0
    
    for filename in file_pairs:
        try:
            # Process DIC image
            dic_path = os.path.join(dic_folder, filename)
            if os.path.exists(dic_path):
                dic_image = imread(dic_path)
                
                print(f"Processing {filename}: Original dtype={dic_image.dtype}, shape={dic_image.shape}, "
                      f"range=[{dic_image.min()}, {dic_image.max()}]")
                
                # Normalize 16-bit images to 8-bit [0, 255] range
                dic_image = normalize_16bit_to_8bit(
                    dic_image, 
                    percentile_normalization=percentile_norm,
                    lower_percentile=lower_perc,
                    upper_percentile=upper_perc
                )
                
                print(f"  -> Normalized: dtype={dic_image.dtype}, range=[{dic_image.min()}, {dic_image.max()}]")
                
                # Save DIC image
                output_dic_path = os.path.join(output_dic, filename)
                imwrite(output_dic_path, dic_image)
                
                # Process mask
                mask_path = os.path.join(mask_folder, filename)
                if os.path.exists(mask_path):
                    processed_mask = process_mask(mask_path)
                    
                    # Save processed mask
                    output_mask_path = os.path.join(output_mask, filename)
                    imwrite(output_mask_path, processed_mask)
                    
                    processed_count += 1
                    print(f"Processed {processed_count}: {filename}")
                else:
                    print(f"Warning: Mask not found for {filename}")
            else:
                print(f"Warning: DIC image not found for {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return processed_count


def create_data_split(file_pairs, train_ratio=0.8, val_ratio=0.2, random_state=42):
    """Split data into train and validation sets."""
    if train_ratio + val_ratio != 1.0:
        raise ValueError("train_ratio + val_ratio must equal 1.0")
    
    train_files, val_files = train_test_split(
        file_pairs, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="Preprocess DIC and mask data for SAM fine-tuning")
    parser.add_argument("--dic_folder", type=str, default="DIC", 
                      help="Path to DIC images folder")
    parser.add_argument("--mask_folder", type=str, default="DIC_mask", 
                      help="Path to mask images folder")
    parser.add_argument("--output_folder", type=str, default="processed_data", 
                      help="Output folder for processed data")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                      help="Ratio of data for training (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                      help="Ratio of data for validation (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, 
                      help="Random state for reproducible splits")
    
    # Normalization arguments for 16-bit data
    parser.add_argument("--percentile_normalization", action="store_true", default=True,
                      help="Use percentile-based normalization for better contrast")
    parser.add_argument("--no_percentile_normalization", dest="percentile_normalization", 
                      action="store_false",
                      help="Use simple min-max normalization instead")
    parser.add_argument("--lower_percentile", type=float, default=1.0,
                      help="Lower percentile for clipping (default: 1.0)")
    parser.add_argument("--upper_percentile", type=float, default=99.0,
                      help="Upper percentile for clipping (default: 99.0)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_ratio + args.val_ratio != 1.0:
        print("Error: train_ratio + val_ratio must equal 1.0")
        return
    
    # Check input folders exist
    if not os.path.exists(args.dic_folder):
        print(f"Error: DIC folder '{args.dic_folder}' does not exist")
        return
    
    if not os.path.exists(args.mask_folder):
        print(f"Error: Mask folder '{args.mask_folder}' does not exist")
        return
    
    print(f"Processing data from:")
    print(f"  DIC folder: {args.dic_folder}")
    print(f"  Mask folder: {args.mask_folder}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Train/Val split: {args.train_ratio:.1f}/{args.val_ratio:.1f}")
    print(f"  Normalization: {'Percentile' if args.percentile_normalization else 'Min-Max'}")
    if args.percentile_normalization:
        print(f"  Percentile range: {args.lower_percentile}% - {args.upper_percentile}%")
    
    # Get file pairs
    file_pairs = get_file_pairs(args.dic_folder, args.mask_folder)
    
    if len(file_pairs) == 0:
        print("Error: No matching file pairs found!")
        return
    
    # Split data
    train_files, val_files = create_data_split(
        file_pairs, args.train_ratio, args.val_ratio, args.random_state
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_files)} files")
    print(f"  Validation: {len(val_files)} files")
    
    # Create output directory structure
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process training data
    print(f"\nProcessing training data...")
    train_count = copy_and_process_data(
        train_files, args.dic_folder, args.mask_folder, args.output_folder, 'train',
        percentile_norm=args.percentile_normalization,
        lower_perc=args.lower_percentile,
        upper_perc=args.upper_percentile
    )
    
    # Process validation data
    print(f"\nProcessing validation data...")
    val_count = copy_and_process_data(
        val_files, args.dic_folder, args.mask_folder, args.output_folder, 'val',
        percentile_norm=args.percentile_normalization,
        lower_perc=args.lower_percentile,
        upper_perc=args.upper_percentile
    )
    
    print(f"\nData preprocessing complete!")
    print(f"  Processed {train_count} training samples")
    print(f"  Processed {val_count} validation samples")
    print(f"  Output directory: {args.output_folder}")
    print(f"\nDirectory structure:")
    print(f"  {args.output_folder}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   └── masks/")
    print(f"  └── val/")
    print(f"      ├── images/")
    print(f"      └── masks/")


if __name__ == "__main__":
    main()