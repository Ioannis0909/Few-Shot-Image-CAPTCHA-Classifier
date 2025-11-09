"""
MODEL.py - Model Architecture and Training
Contains the model definition and training loop.
"""

from networkx import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from typing import Tuple, Dict, List
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from torchvision.models import ViT_B_16_Weights
from PREP import EpisodeSampler


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EmbeddingNetwork(nn.Module):
    """
    ResNet50 backbone with frozen layers and trainable final projection.
    Returns embedding_dim-D L2-normalized embeddings.
    """

    def __init__(self, pretrained: bool = True, embedding_dim: int = 512, freeze_backbone: bool = True):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            embedding_dim: Dimension of output embeddings
            freeze_backbone: If True, freeze all ResNet layers (only train final projection)
        """
        super().__init__()
        
        if pretrained:
            vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            vit = models.vit_b_16(weights=None)

        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        # Remove classification head - use [CLS] token features directly
        vit.heads = nn.Identity()
        self.encoder = vit

        # Freeze all ViT layers if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Frozen all ViT-B/16 backbone layers")

        # Add trainable projection layer: 768 -> embedding_dim
        self.projection = nn.Linear(768, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with frozen backbone and trainable projection.
        Returns embedding_dim-D L2-normalized embeddings.
        """
        # Frozen ViT feature extraction
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.encoder(x)  # (batch, 768) - [CLS] token, already flat!
        else:
            x = self.encoder(x)  # (batch, 768)
        
        # Trainable projection
        x = self.projection(x)  # (batch, embedding_dim)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network using EmbeddingNetwork (fixed ResNet50 backbone).
    """ 
    
    def __init__(self, embedding_network: EmbeddingNetwork):
        super().__init__()
        self.embedding_network = embedding_network
        
    def compute_prototypes(self, support_embeddings: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        """
        Compute class prototypes from support embeddings.
        
        Args:
            support_embeddings: (n_way * k_shot, embedding_dim)
            n_way: Number of classes
            k_shot: Support examples per class
        
        Returns:
            prototypes: (n_way, embedding_dim)
        """
        # Reshape and compute mean per class
        embedding_dim = support_embeddings.size(-1)
        support_embeddings = support_embeddings.view(n_way, k_shot, embedding_dim)
        prototypes = support_embeddings.mean(dim=1)
        return prototypes
    
    def forward(self, support_images: torch.Tensor, query_images: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        """
        Forward pass for one episode.
        
        Args:
            support_images: (n_way * k_shot, 3, H, W)
            query_images: (n_query, 3, H, W)
            n_way: Number of classes
            k_shot: Support examples per class
        
        Returns:
            logits: (n_query, n_way)
        """
        # Embed support and query
        support_embeddings = self.embedding_network(support_images)
        query_embeddings = self.embedding_network(query_images)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, n_way, k_shot)
        
        # Compute Euclidean distances
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        
        # Convert to logits (negative distances)
        return -distances


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters - modify these as needed."""
    
    def __init__(self):
        # Training parameters
        self.n_epochs = 100
        self.learning_rate = 0.001  # Higher LR for training only final layer
        self.patience = 10  # Early stopping patience
        
        # Episode parameters
        self.n_way = 3
        self.k_shot = 5
        self.q_query = 15
        self.episodes_per_epoch = 500
        self.val_episodes_per_epoch = 100
                
        # GPU Optimization parameters
        self.num_workers = 8  # Parallel data loading workers (increase for more cores)
        self.mixed_precision = True  # Use automatic mixed precision (AMP) for faster training

        self.prefetch_episodes = 16
        self.pin_memory = True
        
        # System parameters
        self.device = 'cuda'  # 'cuda' for VM (Only can be run on GPU in current state, but with some changes can be made to run on CPU (terrible idea tho) or on macbook mps but since we have VM we kept it as cuda (i.e. GPU))
        self.save_dir = './outputs/checkpoints'
        
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


class MetricTracker:
    """Track and compute running metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.values = []
    
    def update(self, value: float):
        self.values.append(value)
    
    def average(self) -> float:
        if len(self.values) == 0:
            return 0.0
        return np.mean(self.values)
    
    def std(self) -> float:
        if len(self.values) == 0:
            return 0.0
        return np.std(self.values)


# ============================================================================
# EPISODE TRAINING/EVALUATION
# ============================================================================

def train_episode(model: PrototypicalNetwork, 
                 support_images: torch.Tensor,
                 query_images: torch.Tensor,
                 query_labels: torch.Tensor,
                 n_way: int,
                 k_shot: int,
                 optimizer: optim.Optimizer,
                 device: str,
                 scaler: torch.cuda.amp.GradScaler = None,
                 use_amp: bool = False) -> Tuple[float, float]:
    """Train on one episode with optional mixed precision."""
    model.train()
    
    # Move to device with non_blocking for speed
    support_images = support_images.to(device, non_blocking=True)
    query_images = query_images.to(device, non_blocking=True)
    query_labels = query_labels.to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
    
    # Use automatic mixed precision if enabled
    if use_amp and scaler is not None:
        with torch.amp.autocast('cuda'):
            logits = model(support_images, query_images, n_way, k_shot)
            loss = F.cross_entropy(logits, query_labels)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard forward/backward
        logits = model(support_images, query_images, n_way, k_shot)
        loss = F.cross_entropy(logits, query_labels)
        loss.backward()
        optimizer.step()
    
    # Compute accuracy
    with torch.no_grad():
        accuracy = compute_accuracy(logits, query_labels)
    
    return loss.item(), accuracy


def evaluate_episode(model: PrototypicalNetwork,
                    support_images: torch.Tensor,
                    query_images: torch.Tensor,
                    query_labels: torch.Tensor,
                    n_way: int,
                    k_shot: int,
                    device: str) -> Tuple[float, float]:
    """Evaluate on one episode."""
    model.eval()
    
    with torch.no_grad():
        # Move to device with non_blocking for speed
        support_images = support_images.to(device, non_blocking=True)
        query_images = query_images.to(device, non_blocking=True)
        query_labels = query_labels.to(device, non_blocking=True)
        
        # Forward pass
        logits = model(support_images, query_images, n_way, k_shot)
        
        # Compute loss and accuracy
        loss = F.cross_entropy(logits, query_labels)
        accuracy = compute_accuracy(logits, query_labels)
    
    return loss.item(), accuracy


# ============================================================================
# EPOCH TRAINING/VALIDATION
# ============================================================================

def train_epoch(model: PrototypicalNetwork,
               train_sampler: EpisodeSampler,
               optimizer: optim.Optimizer,
               config: TrainingConfig,
               scaler: torch.cuda.amp.GradScaler = None,
               verbose: bool = True) -> Dict[str, float]:
    """Train for one epoch with GPU optimizations."""
    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()
    
    iterator = tqdm(train_sampler, desc="Training") if verbose else train_sampler
    
    use_amp = config.mixed_precision and scaler is not None
    
    for support_imgs, query_imgs, query_lbls in iterator:
        try:
            loss, acc = train_episode(
                model, support_imgs, query_imgs, query_lbls,
                config.n_way, config.k_shot, optimizer, config.device,
                scaler=scaler, use_amp=use_amp
            )
            
            loss_tracker.update(loss)
            acc_tracker.update(acc)
            
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.4f}'
                })
        except Exception as e:
            print(f"\nError in training episode: {e}")
            print("Skipping this episode...")
            continue
    
    return {
        'loss': loss_tracker.average(),
        'accuracy': acc_tracker.average()
    }


def validate_epoch(model: PrototypicalNetwork,
                  val_sampler: EpisodeSampler,
                  config: TrainingConfig,
                  verbose: bool = True) -> Dict[str, float]:
    """Validate for one epoch."""
    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()
    
    iterator = tqdm(val_sampler, desc="Validation") if verbose else val_sampler
    
    episode_num = 0
    for support_imgs, query_imgs, query_lbls in iterator:
        try:
            episode_num += 1
            loss, acc = evaluate_episode(
                model, support_imgs, query_imgs, query_lbls,
                config.n_way, config.k_shot, config.device
            )
            
            loss_tracker.update(loss)
            acc_tracker.update(acc)
            
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.4f}',
                    'ep': episode_num
                })
        except Exception as e:
            print(f"\nError in validation episode {episode_num}: {e}")
            print("Skipping this episode...")
            continue
    
    return {
        'loss': loss_tracker.average(),
        'accuracy': acc_tracker.average()
    }


# ============================================================================
# COMPLETE TRAINING LOOP
# ============================================================================

def train_model(model: PrototypicalNetwork,
               train_sampler: EpisodeSampler,
               val_sampler: EpisodeSampler,
               config: TrainingConfig) -> Dict[str, List[float]]:
    """
    Complete training loop with GPU optimizations.
    
    Returns:
        Training history dictionary
    """
    # Setup
    model.to(config.device)

    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True

    # Print parameter summary
    print_model_parameters(model)

    # Only optimize parameters that require gradients (the projection layer)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.Adam(trainable_params, lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10
    )
    
    # Initialize AMP scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    print(f"\nStarting training on {config.device}")
    print(f"Training setup: {config.n_way}-way, {config.k_shot}-shot")
    print(f"Episodes per epoch: {config.episodes_per_epoch}")
    print(f"Data loading workers: {config.num_workers}")
    if config.device == 'cuda':
        print("GPU optimizations: cuDNN benchmark, non_blocking transfers")
        if config.mixed_precision:
            print(f"Mixed precision: Enabled (faster training, lower memory)")
    print("-" * 60)
    
    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_sampler, optimizer, config, scaler=scaler, verbose=True
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_sampler, config, verbose=True)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            epochs_without_improvement = 0
            
            checkpoint_path = Path(config.save_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config.to_dict(),
                'history': history
            }, checkpoint_path)
            
            print(f"Saved best model with val_acc: {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= config.patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # Save final model
    final_path = Path(config.save_dir) / 'final_model.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['accuracy'],
        'config': config.to_dict(),
        'history': history
    }, final_path)
    
    # Save history
    history_path = Path(config.save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to {config.save_dir}")
    
    return history


# ============================================================================
# MODEL LOADING
# ============================================================================

def print_model_parameters(model: nn.Module):
    """Print summary of trainable vs frozen parameters."""
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    print("\n" + "=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)
    
    print("\nTRAINABLE PARAMETERS:")
    for name, count in trainable_params:
        print(f"  {name}: {count:,}")
    print(f"Total trainable: {sum(c for _, c in trainable_params):,}")
    
    print("\nFROZEN PARAMETERS:")
    for name, count in frozen_params[:5]:  # Show first 5
        print(f"  {name}: {count:,}")
    if len(frozen_params) > 5:
        print(f"  ... and {len(frozen_params) - 5} more frozen layers")
    print(f"Total frozen: {sum(c for _, c in frozen_params):,}")
    
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
    print("=" * 60 + "\n")


def load_model(checkpoint_path: str, device: str = 'cuda', embedding_dim: int = 512) -> PrototypicalNetwork:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        embedding_dim: Embedding dimension (must match trained model)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model with same architecture
    embedding_net = EmbeddingNetwork(
        pretrained=True,  # Load pretrained ViT weights
        embedding_dim=embedding_dim,
        freeze_backbone=True  # Keep backbone frozen
    )
    model = PrototypicalNetwork(embedding_network=embedding_net)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")

    return model