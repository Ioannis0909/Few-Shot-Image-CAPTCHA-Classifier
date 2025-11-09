
#!/usr/bin/env python3
"""
main.py - CAPTCHA Few-Shot Classification with Frozen ImageNet Baseline
"""

import argparse
import torch

from PREP import prepare_novel_data
from MODEL import create_frozen_baseline  # Match your filename
from FEW_SHOT import FewShotConfig, run_few_shot_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='CAPTCHA Few-Shot Classification - Frozen Baseline')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Google_Recaptcha_V2_Images_Dataset/images')
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("CAPTCHA FEW-SHOT CLASSIFICATION - FROZEN BASELINE")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Device: {device}")
    print("=" * 60)

    # Create frozen baseline
    print("\nCreating frozen ImageNet baseline...")
    model = create_frozen_baseline(device=device)
    
    # Prepare novel class data
    novel_dataset = prepare_novel_data(data_root=args.data_root)
    
    # Evaluation config
    eval_config = FewShotConfig()
    eval_config.device = device
    
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