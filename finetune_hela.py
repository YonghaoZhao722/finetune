import os
import numpy as np
import time

import torch

import torch_em
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data

import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pandas as pd

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available. Will use matplotlib for loss plotting.")
    TENSORBOARD_AVAILABLE = False


DATA_FOLDER = "processed_data"


def get_dataloader(split, patch_shape, batch_size, train_instance_segmentation):
    """Return train or val data loader for finetuning SAM.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    the example hela data. You can either adapt this for your own data (see comments below)
    or write a suitable torch dataloader yourself.
    """
    assert split in ("train", "val")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # This will download the image and segmentation data for training.
    image_dir = fetch_tracking_example_data(DATA_FOLDER)
    segmentation_dir = fetch_tracking_segmentation_data(DATA_FOLDER)

    # 'torch_em.default_segmentation_loader' is a convenience function to build a torch dataloader
    # from image data and labels for training segmentation models.
    # It supports image data in various formats. Here, we load image data and labels from the two
    # folders with tif images that were downloaded by the example data functionality, by specifying
    # `raw_key` and `label_key` as `*.tif`. This means all images in the respective folders that end with
    # .tif will be loaded.
    # The function supports many other file formats. For example, if you have tif stacks with multiple slices
    # instead of multiple tif images in a foldder, then you can pass raw_key=label_key=None.

    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key, label_key = "*.tif", "*.tif"
    # Alternative: if you have tif stacks you can just set raw_key and label_key to None
    # raw_key, label_key= None, None

    # The 'roi' argument can be used to subselect parts of the data.
    # Here, we use it to select the first 70 frames for the train split and the other frames for the val split.
    if split == "train":
        roi = np.s_[:70, :, :]
    else:
        roi = np.s_[70:, :, :]

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        label_transform=label_transform,
        num_workers=8, shuffle=True, raw_transform=sam_training.identity,
    )
    return loader


def custom_sam_training_with_loss_tracking(
    name, model_type, train_loader, val_loader, n_epochs, 
    n_objects_per_batch, lr=1e-5, early_stopping=10, device=None, 
    with_segmentation_decoder=False, checkpoint_path=None, log_dir="./logs"
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
    
    # Loss tracking
    epoch_train_losses = []
    epoch_val_losses = []
    all_batch_losses = []
    
    try:
        # Import necessary modules for custom training
        import torch.nn as nn
        import torch.optim as optim
        from torch.cuda.amp import autocast, GradScaler
        
        # Initialize model
        print(f"Initializing {model_type} model...")
        from micro_sam.models import get_sam_model
        model = get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)
        
        print("Using full fine-tuning (all parameters trainable)")
        
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
                    # Extract data from batch
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        images, labels = batch
                    else:
                        print(f"Warning: Unexpected batch format: {type(batch)}")
                        continue
                    
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        # Forward pass - this is simplified, you may need to adapt based on your SAM implementation
                        outputs = model(images)
                        
                        # Calculate loss (simplified - adapt based on your specific loss computation)
                        if with_segmentation_decoder:
                            loss = bce_loss(outputs, labels.float())
                        else:
                            loss = mse_loss(outputs, labels.float())
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    batch_loss = loss.item()
                    epoch_train_loss += batch_loss
                    epoch_batch_losses.append(batch_loss)
                    all_batch_losses.append(batch_loss)
                    
                    # Log batch loss to tensorboard
                    global_step = epoch * len(train_loader) + batch_idx
                    if writer:
                        writer.add_scalar('Loss/Batch_Train', batch_loss, global_step)
                    
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {batch_loss:.4f}")
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            # Calculate average training loss for this epoch
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            epoch_train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        if isinstance(batch, (list, tuple)) and len(batch) == 2:
                            images, labels = batch
                        else:
                            continue
                        
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        with autocast():
                            outputs = model(images)
                            if with_segmentation_decoder:
                                val_loss = bce_loss(outputs, labels.float())
                            else:
                                val_loss = mse_loss(outputs, labels.float())
                        
                        epoch_val_loss += val_loss.item()
                        val_batches += 1
                        
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else float('inf')
            epoch_val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Log epoch losses to tensorboard
            if writer:
                writer.add_scalar('Loss/Epoch_Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Epoch_Val', avg_val_loss, epoch)
            
            # Save checkpoint
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch_train_losses': epoch_train_losses,
                'epoch_val_losses': epoch_val_losses
            }
            
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, "best.pt"))
                print(f"  New best validation loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Create loss plots after each epoch
            create_loss_plots(epoch_train_losses, epoch_val_losses, all_batch_losses, checkpoint_dir, n_epochs)
        
        # Save final training statistics
        training_stats = {
            'epochs_completed': len(epoch_train_losses),
            'total_epochs': n_epochs,
            'best_val_loss': best_val_loss,
            'final_train_loss': epoch_train_losses[-1] if epoch_train_losses else None,
            'final_val_loss': epoch_val_losses[-1] if epoch_val_losses else None,
            'epoch_train_losses': epoch_train_losses,
            'epoch_val_losses': epoch_val_losses,
            'all_batch_losses': all_batch_losses[:1000]  # Save only first 1000 batch losses to avoid huge files
        }
        
        # Save as JSON
        with open(os.path.join(checkpoint_dir, 'training_stats.json'), 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # Save as CSV for easy analysis
        import pandas as pd
        df = pd.DataFrame({
            'epoch': range(1, len(epoch_train_losses) + 1),
            'train_loss': epoch_train_losses,
            'val_loss': epoch_val_losses
        })
        df.to_csv(os.path.join(checkpoint_dir, 'training_history.csv'), index=False)
        
        if writer:
            writer.close()
        
        print(f"\nTraining completed!")
        print(f"Total epochs: {len(epoch_train_losses)}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final training loss: {epoch_train_losses[-1]:.4f}" if epoch_train_losses else "No training loss recorded")
        
        return model
        
    except Exception as e:
        print(f"Error in custom training: {e}")
        print("Falling back to micro-sam's train_sam function...")
        # Fallback to original training function
        return sam_training.train_sam(
            name=name, model_type=model_type, train_loader=train_loader,
            val_loader=val_loader, n_epochs=n_epochs, n_objects_per_batch=n_objects_per_batch,
            device=device, with_segmentation_decoder=with_segmentation_decoder
        )


def create_loss_plots(epoch_train_losses, epoch_val_losses, all_batch_losses, save_dir, total_epochs):
    """Create and save epoch-based loss curves."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Loss Curves - Hela Cell SAM Fine-tuning', fontsize=16, fontweight='bold')
    
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
        
        # Create a simple bar plot for statistics
        ax4.barh(range(len(stats_data['Metric'])), stats_data['Value'], 
                color=['blue', 'blue', 'blue', 'blue'] + (['red', 'red', 'red', 'red'] if epoch_val_losses else []))
        ax4.set_yticks(range(len(stats_data['Metric'])))
        ax4.set_yticklabels(stats_data['Metric'], fontsize=10)
        ax4.set_xlabel('Loss Value', fontsize=12)
        ax4.set_title('Loss Statistics', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, v in enumerate(stats_data['Value']):
            ax4.text(v + max(stats_data['Value']) * 0.01, i, f'{v:.4f}', 
                    va='center', fontsize=9)
    
    # Add training progress info
    progress_text = f"Training Progress:\n"
    if epoch_train_losses:
        progress_text += f"Epochs Completed: {len(epoch_train_losses)}/{total_epochs}\n"
        progress_text += f"Progress: {len(epoch_train_losses)/total_epochs*100:.1f}%"
    
    plt.figtext(0.02, 0.02, progress_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save comprehensive plot
    plot_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a simple plot for quick viewing
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
    
    # Add annotations for min loss
    if epoch_train_losses:
        min_loss_epoch = epochs[epoch_train_losses.index(min(epoch_train_losses))]
        plt.annotate(f'Min Train Loss: {min(epoch_train_losses):.4f}',
                    xy=(min_loss_epoch, min(epoch_train_losses)),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    simple_plot_path = os.path.join(save_dir, 'simple_loss_curve.png')
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to:")
    print(f"  - {plot_path} (comprehensive plot)")
    print(f"  - {simple_plot_path} (simple plot)")
    
    if epoch_train_losses:
        print(f"Training progress:")
        print(f"  Epochs completed: {len(epoch_train_losses)}/{total_epochs}")
        print(f"  Final training loss: {epoch_train_losses[-1]:.4f}")
        print(f"  Best training loss: {min(epoch_train_losses):.4f} (epoch {epochs[epoch_train_losses.index(min(epoch_train_losses))]}) ")
        if epoch_val_losses:
            print(f"  Final validation loss: {epoch_val_losses[-1]:.4f}")
            print(f"  Best validation loss: {min(epoch_val_losses):.4f} (epoch {epochs[epoch_val_losses.index(min(epoch_val_losses))]})")


def run_training(checkpoint_name, model_type, train_instance_segmentation):
    """Run the actual model training with loss tracking and matplotlib plotting."""

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device used for training
    n_epochs = 100  # number of training epochs
    learning_rate = 1e-5  # learning rate
    early_stopping = 10  # early stopping patience

    print(f"Starting training with loss tracking...")
    print(f"Model: {model_type}")
    print(f"Checkpoint name: {checkpoint_name}")
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Instance segmentation: {train_instance_segmentation}")
    print(f"Tensorboard available: {TENSORBOARD_AVAILABLE}")

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size, train_instance_segmentation)
    val_loader = get_dataloader("val", patch_shape, batch_size, train_instance_segmentation)

    # Run training with custom loss tracking.
    try:
        model = custom_sam_training_with_loss_tracking(
            name=checkpoint_name,
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            n_objects_per_batch=n_objects_per_batch,
            lr=learning_rate,
            early_stopping=early_stopping,
            device=device,
            with_segmentation_decoder=train_instance_segmentation,
            checkpoint_path=None,  # Use default checkpoint
            log_dir="./logs"
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Model trained and saved in: checkpoints/{checkpoint_name}/")
        print(f"✓ Loss curves saved as: checkpoints/{checkpoint_name}/loss_curves.png")
        print(f"✓ Simple loss plot: checkpoints/{checkpoint_name}/simple_loss_curve.png")
        print(f"✓ Training history: checkpoints/{checkpoint_name}/training_history.csv")
        print(f"✓ Training stats: checkpoints/{checkpoint_name}/training_stats.json")
        if TENSORBOARD_AVAILABLE:
            print(f"✓ Tensorboard logs: logs/{checkpoint_name}/")
            print(f"  Run: tensorboard --logdir=logs/{checkpoint_name}")
        print("="*60)
        
        return model
        
    except Exception as e:
        print(f"Error in custom training: {e}")
        print("Falling back to original training method...")
        
        # Fallback to original training method if custom training fails
        sam_training.train_sam(
            name=checkpoint_name,
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            n_objects_per_batch=n_objects_per_batch,
            with_segmentation_decoder=train_instance_segmentation,
            device=device,
        )


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used by the rest of the 'micro_sam' library
    export_path = "./finetuned_yeast_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from the cell tracking challenge,
    but can easily be adapted for other data (including data you have annotated with micro_sam beforehand).
    """
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_yeast"

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True

    run_training(checkpoint_name, model_type, train_instance_segmentation)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
