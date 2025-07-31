#!/usr/bin/env python3
"""
LoRA fine-tuning script for SAM on microscopy data.
Supports both vit_b_lm and vit_l_lm models with PEFT methods.
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
import numpy as np
from tifffile import imread, imwrite
import traceback
import pandas as pd

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model, export_custom_qlora_model

# Try to import PEFT utilities
try:
    from peft_sam.util import get_default_peft_kwargs, RawTrafo
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: peft_sam not available. Install it for PEFT functionality.")
    PEFT_AVAILABLE = False

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available. Will use matplotlib for loss plotting.")
    TENSORBOARD_AVAILABLE = False


class DebugMinInstanceSampler(MinInstanceSampler):
    """MinInstanceSampler with debugging capabilities to handle empty masks"""
    
    def __init__(self, debug_dir="./debug_empty_masks", **kwargs):
        super().__init__(**kwargs)
        self.debug_dir = debug_dir
        self.empty_mask_count = 0
        os.makedirs(debug_dir, exist_ok=True)
        
    def __call__(self, dataset, index):
        try:
            # Try the original sampling
            return super().__call__(dataset, index)
        except Exception as e:
            if "No foreground objects were found" in str(e) or "no instances" in str(e).lower():
                self.empty_mask_count += 1
                print(f"DEBUG: Empty mask detected (count: {self.empty_mask_count})")
                
                # Save debug information
                try:
                    self._save_debug_info(dataset, index, e)
                except Exception as debug_error:
                    print(f"DEBUG: Failed to save debug info: {debug_error}")
                
                # Try to find a valid sample by checking nearby indices
                valid_index = self._find_valid_sample(dataset, index)
                if valid_index is not None:
                    print(f"DEBUG: Using alternative sample at index {valid_index}")
                    return super().__call__(dataset, valid_index)
                else:
                    # If we can't find any valid sample, raise the original error
                    raise e
            else:
                # Re-raise other types of errors
                raise e
    
    def _save_debug_info(self, dataset, index, error):
        """Save the problematic image and mask for debugging"""
        try:
            # Get the raw sample without any transforms
            sample = dataset._get_sample(index)
            raw_image, raw_mask = sample
            
            # Convert to numpy if needed
            if hasattr(raw_image, 'numpy'):
                raw_image = raw_image.numpy()
            if hasattr(raw_mask, 'numpy'):
                raw_mask = raw_mask.numpy()
            
            # Save files with timestamp and index
            timestamp = int(time.time())
            debug_prefix = f"empty_mask_{timestamp}_idx{index}"
            
            # Save image
            image_path = os.path.join(self.debug_dir, f"{debug_prefix}_image.tif")
            if raw_image.ndim == 3 and raw_image.shape[0] == 1:
                raw_image = raw_image[0]  # Remove channel dimension for saving
            imwrite(image_path, raw_image.astype(np.uint8))
            
            # Save mask
            mask_path = os.path.join(self.debug_dir, f"{debug_prefix}_mask.tif")
            if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
                raw_mask = raw_mask[0]  # Remove channel dimension for saving
            imwrite(mask_path, raw_mask.astype(np.uint16))
            
            # Save debug info
            info_path = os.path.join(self.debug_dir, f"{debug_prefix}_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Debug Info for Empty Mask\n")
                f.write(f"========================\n")
                f.write(f"Index: {index}\n")
                f.write(f"Error: {error}\n")
                f.write(f"Image shape: {raw_image.shape}\n")
                f.write(f"Mask shape: {raw_mask.shape}\n")
                f.write(f"Mask unique values: {np.unique(raw_mask)}\n")
                f.write(f"Mask max value: {np.max(raw_mask)}\n")
                f.write(f"Mask sum: {np.sum(raw_mask > 0)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
            
            print(f"DEBUG: Saved debug files with prefix {debug_prefix}")
            
        except Exception as save_error:
            print(f"DEBUG: Error saving debug info: {save_error}")
    
    def _find_valid_sample(self, dataset, original_index, max_attempts=50):
        """Try to find a valid sample by checking nearby indices"""
        dataset_len = len(dataset)
        
        for offset in range(1, min(max_attempts, dataset_len)):
            # Try both forward and backward
            for direction in [1, -1]:
                test_index = (original_index + direction * offset) % dataset_len
                try:
                    # Quick check if this sample might be valid
                    sample = dataset._get_sample(test_index)
                    raw_image, raw_mask = sample
                    
                    # Check if mask has any foreground objects
                    if hasattr(raw_mask, 'numpy'):
                        mask_array = raw_mask.numpy()
                    else:
                        mask_array = raw_mask
                    
                    if np.any(mask_array > 0):
                        # This sample looks valid, try to use it
                        super(DebugMinInstanceSampler, self).__call__(dataset, test_index)
                        return test_index
                except:
                    continue
        
        return None


def get_dataloader(data_folder, split, patch_shape, batch_size, train_instance_segmentation, debug_empty_masks=True):
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
        # NOTE: Reduce min_size to avoid conflicts with your preprocessing
        label_transform = PerObjectDistanceTransform(
            distances=True, 
            boundary_distances=True, 
            directed_distances=False,
            foreground=True, 
            instances=True, 
            min_size=5  # Reduced from 25 to avoid double filtering
        )
    else:
        label_transform = torch_em.transform.label.connected_components
    
    # Use RawTrafo for image preprocessing if PEFT is available
    if PEFT_AVAILABLE:
        raw_transform = RawTrafo(
            desired_shape=patch_shape[-2:], 
            triplicate_dims=True, 
            do_padding=False
        )
    else:
        raw_transform = sam_training.identity
    
    # Use debug sampler if requested
    if train_instance_segmentation and debug_empty_masks:
        debug_dir = os.path.join("./debug_empty_masks", split)
        sampler = DebugMinInstanceSampler(debug_dir=debug_dir)
    else:
        sampler = MinInstanceSampler() if train_instance_segmentation else None
    
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
        raw_transform=raw_transform,
        sampler=sampler,
        num_workers=8, 
        shuffle=True
    )
    
    return loader


def custom_sam_training_with_loss_tracking(
    name, model_type, train_loader, val_loader, n_epochs, 
    n_objects_per_batch, lr, early_stopping, device, 
    peft_kwargs, freeze_parts, with_segmentation_decoder, 
    checkpoint_path, log_dir="./logs"
):
    """Custom training function with epoch-based loss tracking and plotting."""
    
    print("Starting custom SAM training with epoch-based loss tracking...")
    
    # Setup logging first
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join("checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tensorboard writer if available
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, name))
    else:
        writer = None
    
    # Import necessary modules for custom training
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    
    # Loss tracking
    epoch_train_losses = []
    epoch_val_losses = []
    all_batch_losses = []
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    try:
        from micro_sam.models import get_sam_model
        model = get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)
        
        # Apply PEFT if specified
        if peft_kwargs:
            print("Applying PEFT modifications...")
            from peft_sam.util import get_peft_sam_model
            model = get_peft_sam_model(model, **peft_kwargs)
        
        # Freeze parts if specified
        if freeze_parts:
            print(f"Freezing: {freeze_parts}")
            if freeze_parts == "image_encoder":
                for param in model.image_encoder.parameters():
                    param.requires_grad = False
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Falling back to micro-sam's train_sam function...")
        return sam_training.train_sam(
            name=name, model_type=model_type, train_loader=train_loader,
            val_loader=val_loader, n_epochs=n_epochs, n_objects_per_batch=n_objects_per_batch,
            lr=lr, early_stopping=early_stopping, device=device,
            peft_kwargs=peft_kwargs, freeze=freeze_parts,
            with_segmentation_decoder=with_segmentation_decoder,
            checkpoint_path=checkpoint_path
        )
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Setup loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training for {n_epochs} epochs...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_batch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                images, labels = batch
                images = images.to(device)
                if isinstance(labels, dict):
                    labels = {k: v.to(device) if torch.is_tensor(v) else v for k, v in labels.items()}
                else:
                    labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    # Forward pass
                    outputs = model(images)
                    
                    # Calculate loss (simplified - you may need to adjust based on actual SAM loss)
                    if isinstance(outputs, dict) and 'pred_masks' in outputs:
                        loss = bce_loss(outputs['pred_masks'], labels)
                    else:
                        # Fallback loss calculation
                        if hasattr(outputs, 'pred_masks'):
                            loss = bce_loss(outputs.pred_masks, labels)
                        else:
                            loss = mse_loss(outputs, labels)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Record loss
                batch_loss = loss.item()
                epoch_batch_losses.append(batch_loss)
                all_batch_losses.append(batch_loss)
                epoch_train_loss += batch_loss
                
                # Log batch progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    avg_loss = sum(epoch_batch_losses[-10:]) / min(10, len(epoch_batch_losses))
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                          f"Loss: {batch_loss:.4f}, "
                          f"Avg(last 10): {avg_loss:.4f}")
                
                # Log to tensorboard
                if writer:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/Batch_Train', batch_loss, global_step)
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Move batch to device
                    images, labels = batch
                    images = images.to(device)
                    if isinstance(labels, dict):
                        labels = {k: v.to(device) if torch.is_tensor(v) else v for k, v in labels.items()}
                    else:
                        labels = labels.to(device)
                    
                    with autocast():
                        # Forward pass
                        outputs = model(images)
                        
                        # Calculate loss
                        if isinstance(outputs, dict) and 'pred_masks' in outputs:
                            loss = bce_loss(outputs['pred_masks'], labels)
                        else:
                            if hasattr(outputs, 'pred_masks'):
                                loss = bce_loss(outputs.pred_masks, labels)
                            else:
                                loss = mse_loss(outputs, labels)
                    
                    epoch_val_loss += loss.item()
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate average validation loss for this epoch
        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_losses.append(avg_val_loss)
        
        # Log epoch results
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if writer:
            writer.add_scalar('Loss/Epoch_Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Epoch_Val', avg_val_loss, epoch)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Create and save loss plots
    print("Creating epoch-based loss plots...")
    create_epoch_loss_plots(epoch_train_losses, epoch_val_losses, all_batch_losses, 
                           checkpoint_dir, n_epochs)
    
    # Create a summary file
    summary_data = {
        'training_completed': True,
        'epochs_completed': len(epoch_train_losses),
        'total_epochs': n_epochs,
        'model_type': model_type,
        'peft_method': 'lora' if peft_kwargs else 'none',
        'n_objects_per_batch': n_objects_per_batch,
        'learning_rate': lr,
        'final_train_loss': epoch_train_losses[-1] if epoch_train_losses else None,
        'final_val_loss': epoch_val_losses[-1] if epoch_val_losses else None,
        'best_val_loss': best_val_loss,
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_losses': epoch_val_losses
    }
    
    summary_file = os.path.join(checkpoint_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save loss data as CSV for further analysis
    loss_df = pd.DataFrame({
        'epoch': range(1, len(epoch_train_losses) + 1),
        'train_loss': epoch_train_losses,
        'val_loss': epoch_val_losses
    })
    loss_df.to_csv(os.path.join(checkpoint_dir, "epoch_losses.csv"), index=False)
    
    if writer:
        writer.close()
    
    print(f"Training completed! Checkpoints and plots saved in: {checkpoint_dir}")
    
    return None


def create_epoch_loss_plots(epoch_train_losses, epoch_val_losses, all_batch_losses, save_dir, total_epochs):
    """Create and save epoch-based loss curves."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Epoch-wise Training and Validation Loss
    ax1 = axes[0, 0]
    epochs = range(1, len(epoch_train_losses) + 1)
    
    ax1.plot(epochs, epoch_train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
    if epoch_val_losses:
        ax1.plot(epochs, epoch_val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(len(epoch_train_losses), 1))
    
    # Plot 2: Training Loss Only (larger view)
    ax2 = axes[0, 1]
    ax2.plot(epochs, epoch_train_losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss per Epoch', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, max(len(epoch_train_losses), 1))
    
    # Plot 3: Batch-wise losses (smoothed for better visualization)
    ax3 = axes[1, 0]
    if all_batch_losses:
        # Calculate moving average for smoothing
        window_size = max(1, len(all_batch_losses) // 100)  # 1% of total batches
        if window_size > 1 and len(all_batch_losses) > window_size:
            smoothed_losses = []
            for i in range(len(all_batch_losses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(all_batch_losses), i + window_size // 2 + 1)
                smoothed_losses.append(sum(all_batch_losses[start_idx:end_idx]) / (end_idx - start_idx))
            
            ax3.plot(range(len(all_batch_losses)), all_batch_losses, 'lightblue', alpha=0.3, 
                    linewidth=0.5, label='Raw Batch Loss')
            ax3.plot(range(len(smoothed_losses)), smoothed_losses, 'blue', linewidth=2, 
                    label=f'Smoothed (window={window_size})')
        else:
            ax3.plot(range(len(all_batch_losses)), all_batch_losses, 'blue', linewidth=1, 
                    label='Batch Loss')
        
        ax3.set_xlabel('Batch Number', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Training Loss per Batch', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss Statistics
    ax4 = axes[1, 1]
    if epoch_train_losses:
        stats_data = {
            'Metric': ['Min Train Loss', 'Max Train Loss', 'Final Train Loss', 'Avg Train Loss'],
            'Value': [
                min(epoch_train_losses),
                max(epoch_train_losses),
                epoch_train_losses[-1],
                sum(epoch_train_losses) / len(epoch_train_losses)
            ]
        }
        
        if epoch_val_losses:
            stats_data['Metric'].extend(['Min Val Loss', 'Max Val Loss', 'Final Val Loss', 'Avg Val Loss'])
            stats_data['Value'].extend([
                min(epoch_val_losses),
                max(epoch_val_losses),
                epoch_val_losses[-1],
                sum(epoch_val_losses) / len(epoch_val_losses)
            ])
        
        # Create a simple text display of statistics
        ax4.axis('off')
        stats_text = "Training Statistics:\n\n"
        for metric, value in zip(stats_data['Metric'], stats_data['Value']):
            stats_text += f"{metric}: {value:.4f}\n"
        
        stats_text += f"\nEpochs Completed: {len(epoch_train_losses)}/{total_epochs}"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'epoch_loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a simplified epoch-only plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_train_losses, 'b-', linewidth=2, marker='o', markersize=6, label='Training Loss')
    if epoch_val_losses:
        plt.plot(epochs, epoch_val_losses, 'r-', linewidth=2, marker='s', markersize=6, label='Validation Loss')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Progress: Loss vs Epoch', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max(len(epoch_train_losses), 1))
    
    # Add some annotations
    if epoch_train_losses:
        min_loss_epoch = epochs[epoch_train_losses.index(min(epoch_train_losses))]
        plt.annotate(f'Min Train Loss: {min(epoch_train_losses):.4f}',
                    xy=(min_loss_epoch, min(epoch_train_losses)),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    simple_plot_path = os.path.join(save_dir, 'simple_epoch_loss_curve.png')
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to:")
    print(f"  - {plot_path}")
    print(f"  - {simple_plot_path}")
    
    # Print summary statistics
    if epoch_train_losses:
        print(f"\nTraining Summary:")
        print(f"  Epochs completed: {len(epoch_train_losses)}/{total_epochs}")
        print(f"  Final training loss: {epoch_train_losses[-1]:.4f}")
        print(f"  Best training loss: {min(epoch_train_losses):.4f} (epoch {epochs[epoch_train_losses.index(min(epoch_train_losses))]}) ")
        
        if epoch_val_losses:
            print(f"  Final validation loss: {epoch_val_losses[-1]:.4f}")
            print(f"  Best validation loss: {min(epoch_val_losses):.4f} (epoch {epochs[epoch_val_losses.index(min(epoch_val_losses))]})")


def run_lora_training(args):
    """Run LoRA fine-tuning."""
    
    # Check if PEFT is available for LoRA training
    if not PEFT_AVAILABLE and args.peft_method != "freeze_encoder":
        raise ImportError("peft_sam is required for LoRA training. Please install it.")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training hyperparameters
    patch_shape = (1, args.patch_size, args.patch_size)
    
    # PEFT configuration
    if args.peft_method == "freeze_encoder":
        freeze_parts = "image_encoder"
        peft_kwargs = {}
        print("Using frozen encoder training")
    elif args.peft_method is None:
        freeze_parts = None
        peft_kwargs = {}
        print("Using full fine-tuning (no PEFT)")
    else:
        freeze_parts = None
        from peft_sam.util import get_peft_kwargs
        peft_kwargs = get_peft_kwargs(
            peft_module=args.peft_method,
            peft_rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        
        print(f"Using PEFT method: {args.peft_method}")
        print(f"PEFT arguments: {peft_kwargs}")
    
    # Checkpoint naming
    if args.peft_method is None:
        checkpoint_name = f"{args.model_type}/full_finetuning/microscopy_sam"
    else:
        checkpoint_name = f"{args.model_type}/{args.peft_method}/microscopy_sam"
    
    if args.checkpoint_name:
        checkpoint_name = args.checkpoint_name
    
    print(f"Checkpoint will be saved as: {checkpoint_name}")
    
    # Get data loaders
    print("Loading data...")
    train_loader = get_dataloader(
        args.data_folder, "train", patch_shape, args.batch_size, args.train_instance_segmentation, debug_empty_masks=True
    )
    val_loader = get_dataloader(
        args.data_folder, "val", patch_shape, args.batch_size, args.train_instance_segmentation, debug_empty_masks=True
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
        peft_kwargs=peft_kwargs,
        freeze_parts=freeze_parts,
        with_segmentation_decoder=args.train_instance_segmentation,
        checkpoint_path=args.checkpoint_path,
        log_dir=args.log_dir
    )
    
    # Export model
    if args.export_path:
        print(f"Exporting model to: {args.export_path}")
        checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
        
        if args.peft_method == "qlora":
            export_custom_qlora_model(
                checkpoint_path=checkpoint_path,
                model_type=args.model_type,
                save_path=args.export_path,
            )
        else:
            export_custom_sam_model(
                checkpoint_path=checkpoint_path,
                model_type=args.model_type,
                save_path=args.export_path,
            )
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SAM for microscopy data")
    
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
    
    # PEFT arguments
    parser.add_argument("--peft_method", type=str, default="lora",
                      choices=[
                          None, "freeze_encoder", "lora", "qlora", "fact", 
                          "attention_tuning", "adaptformer", "bias_tuning", 
                          "layernorm_tuning", "ssf", "late_lora", "late_ft"
                      ],
                      help="PEFT method to use. Use None for full fine-tuning")
    
    # LoRA specific arguments
    parser.add_argument("--lora_rank", type=int, default=None,
                      help="LoRA rank (default: use PEFT default, typically 16)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                      help="LoRA alpha scaling factor (default: use PEFT default, typically 32)")
    parser.add_argument("--lora_dropout", type=float, default=None,
                      help="LoRA dropout rate (default: use PEFT default, typically 0.1)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Training batch size")
    parser.add_argument("--patch_size", type=int, default=512,
                      help="Input patch size (patch_size x patch_size)")
    parser.add_argument("--n_epochs", type=int, default=200,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate")
    parser.add_argument("--n_objects_per_batch", type=int, default=10,
                      help="Number of objects per batch for sampling (reduced from 25 to be less demanding)")
    parser.add_argument("--early_stopping", type=int, default=15,
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
    print("LoRA Fine-tuning Configuration:")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"PEFT method: {args.peft_method}")
    if args.peft_method in ["lora", "qlora"]:
        print(f"LoRA rank: {args.lora_rank if args.lora_rank else 'default'}")
        print(f"LoRA alpha: {args.lora_alpha if args.lora_alpha else 'default'}")
        print(f"LoRA dropout: {args.lora_dropout if args.lora_dropout else 'default'}")
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
    run_lora_training(args)


if __name__ == "__main__":
    main()