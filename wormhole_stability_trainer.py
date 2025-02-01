"""
Wormhole Stability Model Trainer
"""
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

@dataclass
class TrainingConfig:
    """Configuration for wormhole stability training"""
    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_samples: int = 10000
    validation_split: float = 0.2
    checkpoint_dir: str = "checkpoints"
    model_path: str = "wormhole_stability_model.pth"
    
    # Physics constraints
    min_throat_radius: float = 0.1
    max_energy_density: float = 2.0
    stability_threshold: float = 0.8
    coherence_factor: float = 0.95

class WormholeDataset(Dataset):
    """Dataset for wormhole stability training"""
    def __init__(self, states: torch.Tensor, metrics: torch.Tensor):
        self.states = states
        self.metrics = metrics
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.metrics[idx]

class WormholeStabilityModel(nn.Module):
    """Neural network model for wormhole stability prediction"""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.input_size = 4  # throatRadius, energyDensity, fieldStrength, temporalFlow
        self.output_size = 6  # lambda, stability, coherence, wormholeIntegrity, fieldAlignment, temporalCoupling
        
        self.network = nn.Sequential(
            nn.Linear(self.input_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, self.output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class WormholeStabilityTrainer:
    """Trainer for wormhole stability model"""
    def __init__(
        self,
        model: WormholeStabilityModel,
        config: TrainingConfig
    ):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def generate_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic training data based on wormhole physics"""
        states = torch.rand(self.config.num_samples, 4)
        metrics = torch.zeros(self.config.num_samples, 6)
        
        # Apply physics-based constraints
        states[:, 0] *= self.config.min_throat_radius  # throat radius
        states[:, 1] *= self.config.max_energy_density  # energy density
        
        for i in range(self.config.num_samples):
            throat_radius = states[i, 0]
            energy_density = states[i, 1]
            field_strength = states[i, 2]
            temporal_flow = states[i, 3]
            
            # Complex physics-based relationships
            stability = torch.clamp(
                field_strength * energy_density * self.config.stability_threshold,
                0, 1
            )
            coherence = torch.clamp(
                temporal_flow * field_strength * self.config.coherence_factor,
                0, 1
            )
            
            metrics[i] = torch.tensor([
                0.5,  # lambda (constant)
                stability,
                coherence,
                torch.clamp(throat_radius * energy_density, 0, 1),  # integrity
                torch.clamp(field_strength, 0, 1),  # field alignment
                torch.clamp(temporal_flow, 0, 1)  # temporal coupling
            ])
        
        return states, metrics
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        states, metrics = self.generate_synthetic_data()
        
        # Split data
        split_idx = int(len(states) * (1 - self.config.validation_split))
        train_states, val_states = states[:split_idx], states[split_idx:]
        train_metrics, val_metrics = metrics[:split_idx], metrics[split_idx:]
        
        # Create datasets
        train_dataset = WormholeDataset(train_states, train_metrics)
        val_dataset = WormholeDataset(val_states, val_metrics)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for states, metrics in train_loader:
            states, metrics = states.to(self.device), metrics.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(states)
            loss = self.loss_fn(predictions, metrics)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for states, metrics in val_loader:
                states, metrics = states.to(self.device), metrics.to(self.device)
                predictions = self.model(states)
                loss = self.loss_fn(predictions, metrics)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, path)
    
    def train(self):
        """Train the model"""
        train_loader, val_loader = self.prepare_data()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    self.config.model_path
                )
            
            # Save checkpoint every 100 epochs
            if (epoch + 1) % 100 == 0:
                self.save_checkpoint(epoch, val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.show()

def train_model(config: Optional[TrainingConfig] = None) -> WormholeStabilityModel:
    """Train a new wormhole stability model"""
    if config is None:
        config = TrainingConfig()
    
    model = WormholeStabilityModel(config)
    trainer = WormholeStabilityTrainer(model, config)
    
    print(f"Training on device: {trainer.device}")
    trainer.train()
    trainer.plot_training_history()
    
    return model

if __name__ == "__main__":
    # Train model with default configuration
    trained_model = train_model() 