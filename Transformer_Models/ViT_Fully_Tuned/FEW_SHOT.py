"""
FEW_SHOT.py - Few-Shot Evaluation
Tests the trained model on novel classes with K-shot learning.
"""

import torch
import numpy as np
from typing import Dict
from tqdm import tqdm
import json
from pathlib import Path
from torch.utils.data import DataLoader

from PREP import EpisodeSampler, CaptchaDataset
from MODEL import PrototypicalNetwork


# ============================================================================
# FEW-SHOT EVALUATION CONFIGURATION
# ============================================================================

class FewShotConfig:
    """Few-shot evaluation configuration."""
    
    def __init__(self):
        # Evaluation parameters
        self.k_shot_values = [1, 3, 5, 10]  # K-shot settings to test
        self.q_query = 15                   # Query examples per class
        self.n_episodes = 500               # Episodes per k-shot setting
        
        # Data loading parameters
        self.num_workers = 8
        self.pin_memory = True
        self.prefetch_episodes = 16          # Used as DataLoader prefetch_factor
        
        # System parameters
        self.device = 'cuda'                # Overridden from main.py if needed
        self.output_dir = './outputs/few_shot_results'
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ============================================================================
# SINGLE K-SHOT EVALUATION
# ============================================================================

def evaluate_k_shot(model: PrototypicalNetwork,
                   novel_dataset: CaptchaDataset,
                   n_way: int,
                   k_shot: int,
                   config: FewShotConfig) -> Dict[str, float]:
    """
    Evaluate model on novel classes with K-shot learning for a single k_shot.
    """
    device = config.device
    model.eval()
    model.to(device)
    
    # Episode generator
    sampler = EpisodeSampler(
        dataset=novel_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=config.q_query,
        episodes_per_epoch=config.n_episodes,
    )
    
    # DataLoader with pin_memory and prefetch_episodes
    loader = DataLoader(
        sampler,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_episodes,
        persistent_workers=config.num_workers > 0,
    )
    
    accuracies = []
    losses = []
    
    print(f"\nEvaluating {k_shot}-shot on {n_way} novel classes")
    print(f"Running {config.n_episodes} episodes...")
    
    with torch.no_grad():
        for support_imgs, query_imgs, query_lbls in tqdm(loader, desc=f"{k_shot}-shot"):
            support_imgs = support_imgs.to(device, non_blocking=True)
            query_imgs = query_imgs.to(device, non_blocking=True)
            query_lbls = query_lbls.to(device, non_blocking=True)
            
            logits = model(support_imgs, query_imgs, n_way, k_shot)
            loss = torch.nn.functional.cross_entropy(logits, query_lbls)
            
            preds = logits.argmax(dim=1)
            acc = (preds == query_lbls).float().mean().item()
            
            accuracies.append(acc)
            losses.append(loss.item())
    
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    ci_95 = float(1.96 * std_acc / np.sqrt(config.n_episodes))
    mean_loss = float(np.mean(losses))
    
    results = {
        'k_shot': k_shot,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci_95': ci_95,
        'mean_loss': mean_loss,
        'n_episodes': config.n_episodes
    }
    
    print(f"Results: {mean_acc:.4f} +/- {ci_95:.4f} (95% CI)")
    
    return results


# ============================================================================
# MULTIPLE K-SHOT EVALUATION
# ============================================================================

def evaluate_multiple_k_shots(model: PrototypicalNetwork,
                              novel_dataset: CaptchaDataset,
                              config: FewShotConfig) -> Dict[int, Dict[str, float]]:
    """
    Evaluate across multiple k-shot settings.
    """
    n_way = len(novel_dataset.classes)
    all_results = {}
    
    print("=" * 60)
    print("FEW-SHOT EVALUATION ON NOVEL CLASSES")
    print(f"Novel classes: {novel_dataset.classes}")
    print(f"N-way: {n_way}")
    print("=" * 60)
    
    for k_shot in config.k_shot_values:
        results = evaluate_k_shot(
            model=model,
            novel_dataset=novel_dataset,
            n_way=n_way,
            k_shot=k_shot,
            config=config
        )
        all_results[k_shot] = results
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"{'K-shot':<10} {'Accuracy':<20} {'95% CI':<15}")
    print("-" * 60)
    
    for k_shot in config.k_shot_values:
        acc = all_results[k_shot]['mean_accuracy']
        ci = all_results[k_shot]['ci_95']
        print(f"{k_shot:<10} {acc:.4f}{'':<15} +/- {ci:.4f}")
    
    return all_results


# ============================================================================
# PER-CLASS ANALYSIS
# ============================================================================

def evaluate_per_class(model: PrototypicalNetwork,
                       novel_dataset: CaptchaDataset,
                       k_shot: int,
                       config: FewShotConfig) -> Dict[str, Dict[str, float]]:
    """
    Evaluate per-class accuracy on novel classes for a fixed k_shot.
    """
    device = config.device
    model.eval()
    model.to(device)
    
    n_way = len(novel_dataset.classes)
    
    class_correct = {cls: 0 for cls in novel_dataset.classes}
    class_total = {cls: 0 for cls in novel_dataset.classes}
    
    sampler = EpisodeSampler(
        dataset=novel_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=config.q_query,
        episodes_per_epoch=config.n_episodes
    )
    
    loader = DataLoader(
        sampler,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_episodes,
        persistent_workers=config.num_workers > 0,
    )
    
    print(f"\nPer-class analysis ({k_shot}-shot)")
    print(f"Running {config.n_episodes} episodes...")
    
    with torch.no_grad():
        for support_imgs, query_imgs, query_lbls in tqdm(loader, desc="Per-class"):
            support_imgs = support_imgs.to(device, non_blocking=True)
            query_imgs = query_imgs.to(device, non_blocking=True)
            query_lbls = query_lbls.to(device, non_blocking=True)
            
            logits = model(support_imgs, query_imgs, n_way, k_shot)
            preds = logits.argmax(dim=1)
            
            for pred, true_label in zip(preds, query_lbls):
                class_name = novel_dataset.classes[true_label.item()]
                class_total[class_name] += 1
                if pred == true_label:
                    class_correct[class_name] += 1
    
    results = {}
    
    print(f"\n{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)
    
    for class_name in novel_dataset.classes:
        correct = class_correct[class_name]
        total = class_total[class_name]
        acc = correct / total if total > 0 else 0.0
        
        results[class_name] = {
            'correct': correct,
            'total': total,
            'accuracy': acc
        }
        
        print(f"{class_name:<15} {correct:<10} {total:<10} {acc:.4f}")
    
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    print("-" * 50)
    print(f"{'Overall':<15} {total_correct:<10} {total_samples:<10} {overall_acc:.4f}")
    
    results['overall'] = {
        'correct': total_correct,
        'total': total_samples,
        'accuracy': overall_acc
    }
    
    return results


# ============================================================================
# COMPLETE FEW-SHOT EVALUATION
# ============================================================================

def run_few_shot_evaluation(model: PrototypicalNetwork,
                            novel_dataset: CaptchaDataset,
                            config: FewShotConfig) -> Dict:
    """
    Run complete few-shot evaluation pipeline.
    """
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Evaluate multiple k-shot settings
    k_shot_results = evaluate_multiple_k_shots(model, novel_dataset, config)
    
    # Per-class analysis at 5-shot
    print("\n" + "=" * 60)
    print("PER-CLASS ANALYSIS")
    print("=" * 60)
    
    per_class_results = evaluate_per_class(
        model=model,
        novel_dataset=novel_dataset,
        k_shot=5,
        config=config
    )
    
    full_results = {
        'k_shot_results': k_shot_results,
        'per_class_results': per_class_results,
        'config': config.to_dict()
    }
    
    # Save results with safe types
    results_path = Path(config.output_dir) / 'few_shot_results.json'
    with open(results_path, 'w') as f:
        def to_serializable(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return obj
        
        json.dump(full_results, f, indent=2, default=to_serializable)
    
    print(f"\nResults saved to {results_path}")
    
    return full_results


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def evaluate_novel_classes(checkpoint_path: str,
                           novel_dataset: CaptchaDataset,
                           config: FewShotConfig = None) -> Dict:
    """
    Load model and run full few-shot evaluation on novel classes.
    """
    from MODEL import load_model
    
    if config is None:
        config = FewShotConfig()
    
    model = load_model(checkpoint_path, device=config.device)
    
    results = run_few_shot_evaluation(model, novel_dataset, config)
    
    return results