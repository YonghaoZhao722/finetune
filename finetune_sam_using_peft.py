import os
import argparse

import torch

from torch_em.data import MinInstanceSampler, get_data_loader
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model, export_custom_qlora_model

from peft_sam.util import get_default_peft_kwargs, get_peft_kwargs, RawTrafo


# DATA_ROOT = "processed_data"  # No longer needed - using local paths directly


def get_custom_peft_kwargs(args):
    """Get PEFT kwargs based on command line arguments, falling back to defaults where needed."""
    if args.peft_method == "freeze_encoder":
        return {}
    
    # Start with default kwargs
    peft_kwargs = get_default_peft_kwargs(args.peft_method)
    
    # Parse attention_layers_to_update if provided
    attention_layers_to_update = []
    if args.attention_layers_to_update is not None:
        attention_layers_to_update = [int(x.strip()) for x in args.attention_layers_to_update.split(',')]
    
    # Parse update_matrices if provided
    update_matrices = None
    if args.update_matrices is not None:
        update_matrices = [x.strip() for x in args.update_matrices.split(',')]
    
    # Override with custom parameters if provided
    custom_kwargs = {}
    if args.peft_rank is not None:
        custom_kwargs['peft_rank'] = args.peft_rank
    if args.dropout is not None:
        custom_kwargs['dropout'] = args.dropout
    if args.alpha is not None:
        custom_kwargs['alpha'] = args.alpha
    if args.projection_size is not None:
        custom_kwargs['projection_size'] = args.projection_size
    if args.quantize:
        custom_kwargs['quantize'] = True
    if attention_layers_to_update:
        custom_kwargs['attention_layers_to_update'] = attention_layers_to_update
    if update_matrices is not None:
        custom_kwargs['update_matrices'] = update_matrices
    
    # If any custom parameters are provided, rebuild the kwargs
    if custom_kwargs:
        # Extract the base method parameters from default kwargs
        if args.peft_method in ["lora", "qlora"]:
            peft_kwargs = get_peft_kwargs(
                peft_module="lora",
                peft_rank=custom_kwargs.get('peft_rank', 32),
                quantize=custom_kwargs.get('quantize', args.peft_method == "qlora"),
                attention_layers_to_update=custom_kwargs.get('attention_layers_to_update', []),
                update_matrices=custom_kwargs.get('update_matrices', ["q", "v"])
            )
        elif args.peft_method == "fact":
            peft_kwargs = get_peft_kwargs(
                peft_module="fact",
                peft_rank=custom_kwargs.get('peft_rank', 16),
                dropout=custom_kwargs.get('dropout', 0.1)
            )
        elif args.peft_method == "adaptformer":
            peft_kwargs = get_peft_kwargs(
                peft_module="adaptformer",
                alpha=custom_kwargs.get('alpha', "learnable_scalar"),
                dropout=custom_kwargs.get('dropout', None),
                projection_size=custom_kwargs.get('projection_size', 64)
            )
        elif args.peft_method == "late_lora":
            peft_kwargs = get_peft_kwargs(
                peft_module="lora",
                peft_rank=custom_kwargs.get('peft_rank', 32),
                attention_layers_to_update=custom_kwargs.get('attention_layers_to_update', list(range(6, 12))),
                update_matrices=custom_kwargs.get('update_matrices', ["q", "k", "v", "mlp"])
            )
        elif args.peft_method == "late_ft":
            peft_kwargs = get_peft_kwargs(
                peft_module="ClassicalSurgery",
                attention_layers_to_update=custom_kwargs.get('attention_layers_to_update', list(range(6, 12)))
            )
    
    return peft_kwargs


def get_data_loaders(input_path=None):
    # Use local data paths
    image_path = "/Volumes/ExFAT/finetune/processed_data/DIC"
    mask_path = "/Volumes/ExFAT/finetune/processed_data/DIC_mask"
    
    additional_kwargs = {
        "raw_transform": RawTrafo(desired_shape=(512, 512), triplicate_dims=True, do_padding=False),
        "label_transform": PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
        ),
        "sampler": MinInstanceSampler(),
        "shuffle": True,
    }

    # Create train and validation loaders using local data
    # For training: use first 80% of images, for validation: use last 20%
    import glob
    image_files = sorted(glob.glob(os.path.join(image_path, "*")))
    mask_files = sorted(glob.glob(os.path.join(mask_path, "*")))
    
    # Split data: 80% train, 20% validation
    split_idx = int(0.8 * len(image_files))
    
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    val_image_files = image_files[split_idx:]
    val_mask_files = mask_files[split_idx:]
    
    train_loader = get_data_loader(
        raw_paths=train_image_files,
        raw_key=None,  # for image files, not HDF5
        label_paths=train_mask_files,
        label_key=None,  # for image files, not HDF5
        patch_shape=(512, 512),
        batch_size=1,
        **additional_kwargs,
    )
    
    val_loader = get_data_loader(
        raw_paths=val_image_files,
        raw_key=None,  # for image files, not HDF5
        label_paths=val_mask_files,
        label_key=None,  # for image files, not HDF5
        patch_shape=(512, 512),
        batch_size=1,
        **additional_kwargs,
    )
    
    return train_loader, val_loader


def finetune_sam(args):
    """Script for finetuning SAM (using PEFT methods) on microscopy images.
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = "vit_b_lm"  # override this to start training from another model supported by 'micro-sam'.
    checkpoint_path = None  # override this to start training from a custom checkpoint.
    n_objects_per_batch = 20  # this is the number of objects per batch that will be sampled.

    # whether to freeze the entire image encoder.
    if args.peft_method == "freeze_encoder":
        freeze_parts = "image_encoder"
        peft_kwargs = {}
    else:
        freeze_parts = None
        peft_kwargs = get_custom_peft_kwargs(args)

    # specify checkpoint path depending on the type of finetuning
    if args.peft_method is None:
        checkpoint_name = f"{model_type}/full_finetuning/orgasegment_sam"
    else:
        checkpoint_name = f"{model_type}/{args.peft_method}/orgasegment_sam"

    # all the stuff we need for training
    train_loader, val_loader = get_data_loaders()
    print("PEFT arguments: ", peft_kwargs)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=15,
        lr=5e-6,
        n_epochs=200,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,  # override this to freeze different parts of the model
        device=device,
        peft_kwargs=peft_kwargs,
        with_segmentation_decoder=True,
    )

    # Exports the finetuned PEFT model weights in desired format.
    export_path = None  # override this if you would like to store exportedmodel checkpoints to a desired location.
    if export_path is not None:
        checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
        export_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=export_path,
        )

    # Exports the finetuned QLoRA model weights in desired format.
    export_path = None  # override this if you would like to store exported model checkpoints to a desired location.
    if args.peft_method == "qlora":
        checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
        export_custom_qlora_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for microscopy data.")
    parser.add_argument(
        "--peft_method", type=str, default="lora", help="The method to use for PEFT.",
        choices=[
            "freeze_encoder", "lora", "qlora", "fact", "attention_tuning",
            "adaptformer", "bias_tuning", "layernorm_tuning", "ssf", "late_lora", "late_ft"
        ],
    )
    
    # PEFT method specific parameters
    parser.add_argument("--peft_rank", type=int, default=None, help="Rank for PEFT methods like LoRA, FacT")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate for FacT and AdaptFormer")
    parser.add_argument("--alpha", type=str, default=None, help="Alpha parameter for AdaptFormer (can be 'learnable_scalar' or a float)")
    parser.add_argument("--projection_size", type=int, default=None, help="Projection size for AdaptFormer")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization for QLoRA")
    parser.add_argument("--attention_layers_to_update", type=str, default=None, 
                       help="Comma-separated list of attention layer indices to update (e.g., '6,7,8,9,10,11')")
    parser.add_argument("--update_matrices", type=str, default=None,
                       help="Comma-separated list of matrices to update (e.g., 'q,k,v,mlp')")
    
    args = parser.parse_args()
    finetune_sam(args)


if __name__ == "__main__":
    main()
