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
from skimage.measure import label, regionprops
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


def analyze_instance_sizes(mask):
    """
    Analyze instance sizes in a mask.
    Returns list of instance areas and other statistics.
    """
    if len(np.unique(mask)) <= 1:
        # No instances found
        return []
    
    # Get region properties
    regions = regionprops(mask)
    areas = [region.area for region in regions]
    
    return areas


def process_mask(mask_path, min_instance_size=0):
    """
    Process mask to ensure proper instance segmentation format.
    Each object should have a unique ID, background should be 0.
    Optionally filter out instances smaller than min_instance_size.
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
    
    # Apply instance size filtering if specified
    if min_instance_size > 0:
        filtered_mask, _ = filter_instances_by_size(labeled_mask, min_instance_size)
        return filtered_mask
    
    return labeled_mask.astype(np.uint16)


def filter_instances_by_size(mask, min_instance_size):
    """
    Remove instances from mask that are smaller than minimum size.
    Returns filtered mask and filtering statistics.
    """
    if len(np.unique(mask)) <= 1:
        # No instances found
        return mask, {'removed_count': 0, 'kept_count': 0, 'removed_areas': [], 'kept_areas': []}
    
    # Get region properties
    regions = regionprops(mask)
    
    # Create new mask with only valid instances
    filtered_mask = np.zeros_like(mask)
    new_label = 1
    removed_areas = []
    kept_areas = []
    
    for region in regions:
        if region.area >= min_instance_size:
            # Keep this instance with new consecutive label
            filtered_mask[mask == region.label] = new_label
            kept_areas.append(region.area)
            new_label += 1
        else:
            # Remove this instance
            removed_areas.append(region.area)
    
    stats = {
        'removed_count': len(removed_areas),
        'kept_count': len(kept_areas),
        'removed_areas': removed_areas,
        'kept_areas': kept_areas
    }
    
    return filtered_mask.astype(np.uint16), stats


def analyze_and_filter_instances(file_pairs, mask_folder, min_instance_size):
    """
    Analyze instance sizes and return filtering statistics.
    Returns tuple of (valid_pairs, filtering_info).
    """
    valid_pairs = []
    filtering_info = []
    
    print(f"\nFiltering instances with minimum size: {min_instance_size} pixels")
    print("=" * 60)
    
    for filename in file_pairs:
        mask_path = os.path.join(mask_folder, filename)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {filename}, skipping")
            continue
            
        try:
            # Load and process mask (without filtering first to get original stats)
            original_mask = process_mask(mask_path, min_instance_size=0)
            
            # Analyze and filter instances
            filtered_mask, stats = filter_instances_by_size(original_mask, min_instance_size)
            
            if stats['kept_count'] == 0:
                # No valid instances remain
                filtering_info.append({
                    'filename': filename,
                    'status': 'excluded',
                    'original_count': stats['removed_count'],
                    'kept_count': 0,
                    'removed_count': stats['removed_count'],
                    'removed_areas': stats['removed_areas'],
                    'kept_areas': []
                })
                print(f"EXCLUDED: {filename} - All {stats['removed_count']} instances too small (areas: {stats['removed_areas']})")
            else:
                # Some instances remain
                valid_pairs.append(filename)
                filtering_info.append({
                    'filename': filename,
                    'status': 'kept',
                    'original_count': stats['kept_count'] + stats['removed_count'],
                    'kept_count': stats['kept_count'],
                    'removed_count': stats['removed_count'],
                    'removed_areas': stats['removed_areas'],
                    'kept_areas': stats['kept_areas']
                })
                
                if stats['removed_count'] > 0:
                    print(f"MODIFIED: {filename} - Kept {stats['kept_count']}/{stats['kept_count'] + stats['removed_count']} instances, "
                          f"removed {stats['removed_count']} small instances")
                else:
                    print(f"KEPT: {filename} - All {stats['kept_count']} instances valid")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            filtering_info.append({
                'filename': filename,
                'status': 'error',
                'original_count': 0,
                'kept_count': 0,
                'removed_count': 0,
                'removed_areas': [],
                'kept_areas': [],
                'error': str(e)
            })
    
    return valid_pairs, filtering_info


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
                         percentile_norm=True, lower_perc=1, upper_perc=99, min_instance_size=0):
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
                    processed_mask = process_mask(mask_path, min_instance_size)
                    
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


def create_microsam_structure(file_pairs, dic_folder, mask_folder, output_folder,
                             percentile_norm=True, lower_perc=1, upper_perc=99, min_instance_size=0):
    """
    Create micro_sam compatible folder structure for direct use with finetune_hela.py.
    
    """
    # Create micro_sam compatible paths
    images_output_dir = os.path.join(output_folder, "DIC")
    masks_output_dir = os.path.join(output_folder, "DIC_mask")
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    
    processed_count = 0
    
    print(f"Creating micro_sam compatible structure:")
    print(f"  Images will be saved to: {images_output_dir}")
    print(f"  Masks will be saved to: {masks_output_dir}")
    print(f"  Files will be renamed to t000.tif, t001.tif, etc.")
    
    for idx, filename in enumerate(file_pairs):
        try:
            # Generate micro_sam compatible filename (t000.tif, t001.tif, etc.)
            new_filename = f"t{str(idx).zfill(3)}.tif"
            
            # Process DIC image
            dic_path = os.path.join(dic_folder, filename)
            if os.path.exists(dic_path):
                dic_image = imread(dic_path)
                
                print(f"Processing {filename} -> {new_filename}: Original dtype={dic_image.dtype}, shape={dic_image.shape}, "
                      f"range=[{dic_image.min()}, {dic_image.max()}]")
                
                # Normalize 16-bit images to 8-bit [0, 255] range
                dic_image = normalize_16bit_to_8bit(
                    dic_image, 
                    percentile_normalization=percentile_norm,
                    lower_percentile=lower_perc,
                    upper_percentile=upper_perc
                )
                
                print(f"  -> Normalized: dtype={dic_image.dtype}, range=[{dic_image.min()}, {dic_image.max()}]")
                
                # Save DIC image with micro_sam naming
                output_image_path = os.path.join(images_output_dir, new_filename)
                imwrite(output_image_path, dic_image)
                
                # Process mask
                mask_path = os.path.join(mask_folder, filename)
                if os.path.exists(mask_path):
                    processed_mask = process_mask(mask_path, min_instance_size)
                    
                    # Save processed mask with micro_sam naming
                    output_mask_path = os.path.join(masks_output_dir, new_filename)
                    imwrite(output_mask_path, processed_mask)
                    
                    processed_count += 1
                    print(f"Processed {processed_count}: {filename} -> {new_filename}")
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


def detect_microsam_structure(data_folder):
    """
    Detect and resolve the micro_sam official download structure.
    
    Expected structure:
    - Images: data_folder/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01/
    - Masks: data_folder/hela-ctc-01-gt.zip.unzip/masks/
    
    Returns:
        tuple: (dic_folder, mask_folder) or (None, None) if not found
    """
    data_path = Path(data_folder)
    
    # Look for the image folder
    dic_candidates = [
        data_path / "DIC-C2DH-HeLa.zip.unzip" / "DIC-C2DH-HeLa" / "01",
        data_path / "DIC-C2DH-HeLa.zip.unzip" / "DIC-C2DH-HeLa" / "01_RES",
        data_path / "DIC-C2DH-HeLa.zip.unzip" / "DIC-C2DH-HeLa" / "01_GT" / "SEG",
    ]
    
    # Look for the mask folder  
    mask_candidates = [
        data_path / "hela-ctc-01-gt.zip.unzip" / "masks",
        data_path / "DIC-C2DH-HeLa.zip.unzip" / "DIC-C2DH-HeLa" / "01_GT" / "SEG",
        data_path / "DIC-C2DH-HeLa.zip.unzip" / "DIC-C2DH-HeLa" / "01_ST" / "SEG",
    ]
    
    dic_folder = None
    mask_folder = None
    
    # Find existing DIC folder
    for candidate in dic_candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it contains .tif files
            tif_files = list(candidate.glob("*.tif")) + list(candidate.glob("*.tiff"))
            if tif_files:
                dic_folder = str(candidate)
                print(f"Found DIC images in: {dic_folder}")
                print(f"  Contains {len(tif_files)} .tif files")
                break
    
    # Find existing mask folder
    for candidate in mask_candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it contains .tif files
            tif_files = list(candidate.glob("*.tif")) + list(candidate.glob("*.tiff"))
            if tif_files:
                mask_folder = str(candidate)
                print(f"Found mask images in: {mask_folder}")
                print(f"  Contains {len(tif_files)} .tif files")
                break
    
    return dic_folder, mask_folder


def resolve_paths(dic_folder, mask_folder, data_folder, use_microsam_data):
    """
    Resolve the actual paths for DIC and mask folders.
    
    Args:
        dic_folder: User-specified DIC folder (or None)
        mask_folder: User-specified mask folder (or None)
        data_folder: Base data folder
        use_microsam_data: Whether to auto-detect micro_sam structure
    
    Returns:
        tuple: (resolved_dic_folder, resolved_mask_folder)
    """
    resolved_dic = dic_folder
    resolved_mask = mask_folder
    
    if use_microsam_data:
        print("Auto-detecting micro_sam download structure...")
        detected_dic, detected_mask = detect_microsam_structure(data_folder)
        
        if detected_dic and not resolved_dic:
            resolved_dic = detected_dic
            print(f"Using auto-detected DIC folder: {resolved_dic}")
        
        if detected_mask and not resolved_mask:
            resolved_mask = detected_mask  
            print(f"Using auto-detected mask folder: {resolved_mask}")
        
        if not detected_dic and not detected_mask:
            print("Warning: Could not auto-detect micro_sam structure")
            print("Expected structure:")
            print(f"  - Images: {data_folder}/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01/")
            print(f"  - Masks: {data_folder}/hela-ctc-01-gt.zip.unzip/masks/")
            print("Please specify --dic_folder and --mask_folder manually.")
    
    return resolved_dic, resolved_mask


def main():
    parser = argparse.ArgumentParser(description="Preprocess DIC and mask data for SAM fine-tuning")
    
    # Data source options
    parser.add_argument("--use_microsam_data", action="store_true", 
                      help="Auto-detect and use micro_sam official download structure")
    parser.add_argument("--create_microsam_structure", action="store_true",
                      help="Create output in micro_sam compatible structure (for use with finetune_hela.py)")
    parser.add_argument("--data_folder", type=str, default="processed_data",
                      help="Base folder containing downloaded data (used with --use_microsam_data)")
    
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
    
    # Instance size filtering
    parser.add_argument("--min_instance_size", type=int, default=0,
                      help="Minimum instance size in pixels (default: 0, no filtering)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_ratio + args.val_ratio != 1.0:
        print("Error: train_ratio + val_ratio must equal 1.0")
        return
    
    # Resolve input folder paths
    dic_folder, mask_folder = resolve_paths(
        args.dic_folder, args.mask_folder, args.data_folder, args.use_microsam_data
    )
    
    # Validate that we have resolved paths
    if not dic_folder and not mask_folder:
        print("Error: Could not determine DIC and mask folders")
        if args.use_microsam_data:
            print("Ensure you have downloaded data using micro_sam, or specify paths manually")
        else:
            print("Please specify --dic_folder and --mask_folder")
        return
    
    if not dic_folder:
        print("Error: Could not determine DIC folder")
        print("Please specify --dic_folder manually")
        return
        
    if not mask_folder:
        print("Error: Could not determine mask folder") 
        print("Please specify --mask_folder manually")
        return
    
    # Check that resolved folders exist
    if not os.path.exists(dic_folder):
        print(f"Error: DIC folder '{dic_folder}' does not exist")
        return
    
    if not os.path.exists(mask_folder):
        print(f"Error: Mask folder '{mask_folder}' does not exist")
        return
    
    print(f"Processing data from:")
    print(f"  DIC folder: {dic_folder}")
    print(f"  Mask folder: {mask_folder}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Train/Val split: {args.train_ratio:.1f}/{args.val_ratio:.1f}")
    print(f"  Normalization: {'Percentile' if args.percentile_normalization else 'Min-Max'}")
    if args.percentile_normalization:
        print(f"  Percentile range: {args.lower_percentile}% - {args.upper_percentile}%")
    print(f"  Minimum instance size: {args.min_instance_size} pixels")
    print(f"  Using micro_sam data: {args.use_microsam_data}")
    
    # Get file pairs
    file_pairs = get_file_pairs(dic_folder, mask_folder)
    
    if len(file_pairs) == 0:
        print("Error: No matching file pairs found!")
        return
    
    # Apply instance size filtering if specified
    filtering_info = []
    if args.min_instance_size > 0:
        file_pairs, filtering_info = analyze_and_filter_instances(
            file_pairs, mask_folder, args.min_instance_size
        )
        
        if len(file_pairs) == 0:
            print("Error: No files remain after filtering!")
            return
    
    # Create output directory structure
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.create_microsam_structure:
        # Create micro_sam compatible structure (no train/val split - handled by finetune_hela.py)
        print(f"\nCreating micro_sam compatible structure...")
        print(f"Note: Train/Val split will be handled automatically by finetune_hela.py using roi parameter")
        print(f"Total files to process: {len(file_pairs)}")
        
        total_count = create_microsam_structure(
            file_pairs, dic_folder, mask_folder, args.output_folder,
            percentile_norm=args.percentile_normalization,
            lower_perc=args.lower_percentile,
            upper_perc=args.upper_percentile,
            min_instance_size=args.min_instance_size
        )
        
        print(f"\nData preprocessing complete!")
        print(f"  Processed {total_count} samples in micro_sam format")
        print(f"  Output directory: {args.output_folder}")
        
        print(f"\nCreated micro_sam compatible structure:")
        print(f"  {args.output_folder}/")
        print(f"  ├── DIC-C2DH-HeLa.zip.unzip/")
        print(f"  │   └── DIC-C2DH-HeLa/")
        print(f"  │       └── 01/              # Images (t000.tif, t001.tif, ...)")
        print(f"  └── hela-ctc-01-gt.zip.unzip/")
        print(f"      └── masks/               # Masks (t000.tif, t001.tif, ...)")
        
        print(f"\n🎉 Ready for training!")
        print(f"  Now you can run: python finetune_hela.py")
        print(f"  The script will automatically use the created data structure.")
        
    else:
        # Traditional train/val split processing
        # Split data
        train_files, val_files = create_data_split(
            file_pairs, args.train_ratio, args.val_ratio, args.random_state
        )
        
        print(f"\nData split:")
        if args.min_instance_size > 0:
            excluded_count = len([info for info in filtering_info if info['status'] == 'excluded'])
            print(f"  Files after filtering: {len(file_pairs)}")
            print(f"  Files excluded: {excluded_count}")
        print(f"  Training: {len(train_files)} files")
        print(f"  Validation: {len(val_files)} files")
        
        # Process training data
        print(f"\nProcessing training data...")
        train_count = copy_and_process_data(
            train_files, dic_folder, mask_folder, args.output_folder, 'train',
            percentile_norm=args.percentile_normalization,
            lower_perc=args.lower_percentile,
            upper_perc=args.upper_percentile,
            min_instance_size=args.min_instance_size
        )
        
        # Process validation data
        print(f"\nProcessing validation data...")
        val_count = copy_and_process_data(
            val_files, dic_folder, mask_folder, args.output_folder, 'val',
            percentile_norm=args.percentile_normalization,
            lower_perc=args.lower_percentile,
            upper_perc=args.upper_percentile,
            min_instance_size=args.min_instance_size
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
    
    # Report filtering results
    if args.min_instance_size > 0 and filtering_info:
        print(f"\nFiltering Summary:")
        print(f"  Minimum instance size: {args.min_instance_size} pixels")
        
        excluded_files = [info for info in filtering_info if info['status'] == 'excluded']
        modified_files = [info for info in filtering_info if info['status'] == 'kept' and info['removed_count'] > 0]
        kept_files = [info for info in filtering_info if info['status'] == 'kept' and info['removed_count'] == 0]
        
        print(f"  Files completely excluded: {len(excluded_files)}")
        print(f"  Files with instances removed: {len(modified_files)}")
        print(f"  Files kept unchanged: {len(kept_files)}")
        
        if excluded_files:
            print(f"\nCompletely excluded files:")
            print("=" * 80)
            for info in excluded_files:
                print(f"  {info['filename']}")
                print(f"    Original instances: {info['original_count']}")
                print(f"    All instance areas: {info['removed_areas']}")
                print()
        
        if modified_files:
            print(f"\nFiles with instances removed:")
            print("=" * 80)
            for info in modified_files:
                print(f"  {info['filename']}")
                print(f"    Original instances: {info['original_count']}")
                print(f"    Kept instances: {info['kept_count']}")
                print(f"    Removed instances: {info['removed_count']}")
                print(f"    Removed areas: {info['removed_areas']}")
                print(f"    Kept areas: {info['kept_areas']}")
                print()


if __name__ == "__main__":
    main()