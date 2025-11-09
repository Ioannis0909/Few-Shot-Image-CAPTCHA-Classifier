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
    ViT-B/16 backbone without projection layers.
    Returns 768-D L2-normalized embeddings from [CLS] token.
    """

    def __init__(self):
        
        super().__init__()
        
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        self.embedding_dim = 768  # ViT-B/16 hidden dimension
        
        # Remove classification head - use [CLS] token features directly
        vit.heads = nn.Identity()
        self.encoder = vit
        
        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ViT-B/16 backbone without projection layers.
        Returns 768-D L2-normalized embeddings from [CLS] token.
        """
        x = self.encoder(x)  # (batch, 768) - already flat!
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
# MODEL LOADING
# ============================================================================

def create_frozen_baseline(device: str = 'cuda') -> PrototypicalNetwork:
    """Create frozen ImageNet baseline - no training needed."""
    embedding_net = EmbeddingNetwork(pretrained=True)  # Already frozen
    model = PrototypicalNetwork(embedding_network=embedding_net)
    model.to(device)
    model.eval()
    return model