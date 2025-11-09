#!/usr/bin/env python3
"""
main.py - Complete CAPTCHA Few-Shot Classification Pipeline

Usage:
    python main.py --data_root /path/to/images --mode train
    python main.py --data_root /path/to/images --mode eval --checkpoint path/to/best_model.pt
    python main.py --data_root /path/to/images --mode both
"""

import argparse
import torch
from torch.utils.data import DataLoader

from PREP import prepare_base_data, prepare_novel_data, EpisodeSampler
from MODEL import (
    EmbeddingNetwork,
    PrototypicalNetwork,
    TrainingConfig,
    train_model,
    load_model
)
from FEW_SHOT import FewShotConfig, run_few_shot_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='CAPTCHA Few-Shot Classification')

    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to Google_Recaptcha_V2_Images_Dataset/images'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='both',
        choices=['train', 'eval', 'both'],
        help='Mode: train, eval, or both (default: both)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation mode (required for --mode eval)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Enforce GPU-only setup (with fallback if unavailable)
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. This code is optimized for GPU-only runs.")
        device = 'cpu'
    else:
        device = 'cuda'

    print("=" * 60)
    print("CAPTCHA FEW-SHOT CLASSIFICATION")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    if args.mode in ['train', 'both']:
        print("\n" + "=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)

        # Config
        config = TrainingConfig()
        config.device = device

        # Prepare base data
        train_dataset, val_dataset, test_dataset, _ = prepare_base_data(
            data_root=args.data_root,
            num_workers=config.num_workers
        )

        # Episode samplers (logical episode generators)
        train_episode_sampler = EpisodeSampler(
            dataset=train_dataset,
            n_way=config.n_way,
            k_shot=config.k_shot,
            q_query=config.q_query,
            episodes_per_epoch=config.episodes_per_epoch,
        )

        val_episode_sampler = EpisodeSampler(
            dataset=val_dataset,
            n_way=config.n_way,
            k_shot=config.k_shot,
            q_query=config.q_query,
            episodes_per_epoch=config.val_episodes_per_epoch,
        )

        # DataLoaders with pin_memory + prefetch_episodes
        train_loader = DataLoader(
            train_episode_sampler,
            batch_size=None,  # each yield is (support, query, labels)
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_episodes,
            persistent_workers=config.num_workers > 0,
        )

        val_loader = DataLoader(
            val_episode_sampler,
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_episodes,
            persistent_workers=config.num_workers > 0,
        )

        # Model configuration
        embedding_dim = 512  # Can be changed (e.g., 256, 512, 1024)
        
        # Model (frozen ResNet50 backbone with trainable projection layer)
        embedding_net = EmbeddingNetwork(
            pretrained=True,
            embedding_dim=embedding_dim,
        )
        model = PrototypicalNetwork(embedding_network=embedding_net)

        print(f"Embedding dimension: {embedding_net.embedding_dim}")

        # Train
        history = train_model(
            model=model,
            train_sampler=train_loader,
            val_sampler=val_loader,
            config=config
        )

        # Use best checkpoint for downstream eval (if mode == both)
        args.checkpoint = f"{config.save_dir}/best_model.pt"

    # ========================================================================
    # EVALUATION PHASE
    # ========================================================================
    if args.mode in ['eval', 'both']:
        print("\n" + "=" * 60)
        print("FEW-SHOT EVALUATION PHASE")
        print("=" * 60)

        if args.checkpoint is None:
            print("Error: --checkpoint is required for eval mode")
            return

        # Prepare novel class data
        novel_dataset = prepare_novel_data(data_root=args.data_root)

        # Few-shot eval config
        eval_config = FewShotConfig()
        eval_config.device = device
        
        # Model configuration (must match training)
        embedding_dim = 512  # Must match the value used during training

        # Load trained model
        print(f"\nLoading model from {args.checkpoint}")
        model = load_model(args.checkpoint, device=device, embedding_dim=embedding_dim)

        # Run evaluation
        results = run_few_shot_evaluation(
            model=model,
            novel_dataset=novel_dataset,
            config=eval_config
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()