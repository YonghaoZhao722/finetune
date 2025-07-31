#!/usr/bin/env python3
"""
Normal (full) fine-tuning script for SAM on microscopy data.
Supports both vit_b_lm and vit_l_lm models with full parameter updates.
"""

import os
import argparse
import time
import torch
import torch_em
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data import MinInstanceSampler
import matplotlib.pyplot as plt
from collections import defaultdict
import json

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available. Will use matplotlib for loss plotting.")
    TENSORBOARD_AVAILABLE = False


def get_dataloader(data_folder, split, patch_shape, batch_size, train_instance_segmentation):
    """Return train or val data loader for finetuning SAM."""
    assert split in ("train", "val")
    
    # Define paths to processed data
    image_dir = os.path.join(data_folder, split, 'images')
    mask_dir = os.path.join(data_folder, split, 'masks')
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Images directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"Masks directory not found: {mask_dir}")
    
    # Load images from multiple files in folder via pattern
    raw_key, label_key = "*.tif", "*.tif"
    
    if train_instance_segmentation:
        # Use PerObjectDistanceTransform for automatic instance segmentation
        label_transform = PerObjectDistanceTransform(
            distances=True, 
            boundary_distances=True, 
            directed_distances=False,
            foreground=True, 
            instances=True, 
            min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components
    
    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, 
        raw_key=raw_key,
        label_paths=mask_dir, 
        label_key=label_key,
        patch_shape=patch_shape, 
        batch_size=batch_size,
        ndim=2, 
        is_seg_dataset=True,
        label_transform=label_transform,
        raw_transform=sam_training.identity,
        sampler=MinInstanceSampler() if train_instance_segmentation else None,
        num_workers=8, 
        shuffle=True
    )
    
    return loader


def custom_sam_training_with_loss_tracking(
    name, model_type, train_loader, val_loader, n_epochs, 
    n_objects_per_batch, lr, early_stopping, device, 
    with_segmentation_decoder, checkpoint_path, log_dir="./logs"
):
    """Custom training function with batch-wise loss tracking."""
    
    print("Starting training with micro-sam's train_sam function...")
    
    # Setup logging first
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join("checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tensorboard writer if available
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, name))
    else:
        writer = None
    
    try:
        # Use micro-sam's training function directly
        sam_training.train_sam(
            name=name,
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            n_objects_per_batch=n_objects_per_batch,
            lr=lr,
            early_stopping=early_stopping,
            device=device,
            with_segmentation_decoder=with_segmentation_decoder,
            checkpoint_path=checkpoint_path,
        )
    finally:
        if writer:
            writer.close()
    
    # Create a summary file
    summary_data = {
        'training_completed': True,
        'epochs': n_epochs,
        'model_type': model_type,
        'training_method': 'full_finetuning',
        'n_objects_per_batch': n_objects_per_batch,
        'learning_rate': lr
    }
    
    summary_file = os.path.join(checkpoint_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Training completed! Checkpoints saved in: {checkpoint_dir}")
    
    return None  # Model is saved by micro-sam


def create_loss_plots(train_losses, val_losses, batch_losses, save_dir):
    """Create and save loss plots."""
    
    # Plot 1: Epoch-wise losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Batch-wise losses (smoothed)
    plt.subplot(1, 2, 2)
    
    # Smooth batch losses for better visualization
    window_size = max(1, len(batch_losses) // 100)  # Smooth over 1% of batches
    if window_size > 1:
        smoothed_batch_losses = []
        for i in range(len(batch_losses)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(batch_losses), i + window_size // 2 + 1)
            smoothed_batch_losses.append(sum(batch_losses[start_idx:end_idx]) / (end_idx - start_idx))
        
        plt.plot(range(len(batch_losses)), batch_losses, 'lightblue', alpha=0.3, label='Raw Batch Loss')
        plt.plot(range(len(smoothed_batch_losses)), smoothed_batch_losses, 'blue', linewidth=2, label='Smoothed Batch Loss')
    else:
        plt.plot(range(len(batch_losses)), batch_losses, 'blue', linewidth=1, label='Batch Loss')
    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plots saved to: {os.path.join(save_dir, 'loss_curves.png')}")


def run_normal_training(args):
    """Run normal (full) fine-tuning."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training hyperparameters
    patch_shape = (1, args.patch_size, args.patch_size)
    
    # Checkpoint naming
    checkpoint_name = f"{args.model_type}/full_finetuning/microscopy_sam"
    if args.checkpoint_name:
        checkpoint_name = args.checkpoint_name
    
    print(f"Checkpoint will be saved as: {checkpoint_name}")
    
    # Get data loaders
    print("Loading data...")
    train_loader = get_dataloader(
        args.data_folder, "train", patch_shape, args.batch_size, args.train_instance_segmentation
    )
    val_loader = get_dataloader(
        args.data_folder, "val", patch_shape, args.batch_size, args.train_instance_segmentation
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Run training with custom loss tracking
    print("Starting training...")
    model = custom_sam_training_with_loss_tracking(
        name=checkpoint_name,
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        n_objects_per_batch=args.n_objects_per_batch,
        lr=args.learning_rate,
        early_stopping=args.early_stopping,
        device=device,
        with_segmentation_decoder=args.train_instance_segmentation,
        checkpoint_path=args.checkpoint_path,
        log_dir=args.log_dir
    )
    
    # Export model
    if args.export_path:
        print(f"Exporting model to: {args.export_path}")
        checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=args.model_type,
            save_path=args.export_path,
        )
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Normal fine-tune SAM for microscopy data")
    
    # Data arguments
    parser.add_argument("--data_folder", type=str, default="processed_data",
                      help="Path to processed data folder")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="vit_b_lm", 
                      choices=["vit_b", "vit_l", "vit_h", "vit_b_lm", "vit_l_lm", "vit_h_lm"],
                      help="SAM model type to use")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                      help="Path to custom checkpoint to start from")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                      help="Custom name for checkpoint directory")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Training batch size")
    parser.add_argument("--patch_size", type=int, default=512,
                      help="Input patch size (patch_size x patch_size)")
    parser.add_argument("--n_epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                      help="Learning rate (lower for full fine-tuning)")
    parser.add_argument("--n_objects_per_batch", type=int, default=25,
                      help="Number of objects per batch for sampling")
    parser.add_argument("--early_stopping", type=int, default=10,
                      help="Early stopping patience")
    
    # Segmentation arguments
    parser.add_argument("--train_instance_segmentation", action="store_true", default=True,
                      help="Train with instance segmentation decoder")
    parser.add_argument("--no_instance_segmentation", dest="train_instance_segmentation", 
                      action="store_false",
                      help="Disable instance segmentation training")
    
    # Export arguments
    parser.add_argument("--export_path", type=str, default=None,
                      help="Path to export the trained model")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs",
                      help="Directory for logging and tensorboard files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist.")
        print("Please run data_preprocessing.py first to prepare the data.")
        return
    
    train_dir = os.path.join(args.data_folder, "train")
    val_dir = os.path.join(args.data_folder, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Training or validation directories not found in '{args.data_folder}'.")
        print("Please run data_preprocessing.py first to prepare the data.")
        return
    
    print("="*60)
    print("Normal Fine-tuning Configuration:")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Training method: Full fine-tuning (all parameters)")
    print(f"Data folder: {args.data_folder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Objects per batch: {args.n_objects_per_batch}")
    print(f"Instance segmentation: {args.train_instance_segmentation}")
    print(f"Log directory: {args.log_dir}")
    print("="*60)
    
    # Run training
    run_normal_training(args)


if __name__ == "__main__":
    main()