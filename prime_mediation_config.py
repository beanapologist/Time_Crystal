"""
Configuration and utilities for prime-mediated time modulation
"""
from dataclasses import dataclass, field
import numpy as np
from sympy import prime
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import os
from wormhole_stability_trainer import WormholeStabilityTrainer, train_model, TrainingConfig

@dataclass
class WormholeConfig:
    """Configuration for wormhole stability parameters"""
    hidden_size: int = 128
    learning_rate: float = 1e-3
    input_size: int = 4  # throatRadius, energyDensity, fieldStrength, temporalFlow
    output_size: int = 6  # lambda, stability, coherence, wormholeIntegrity, fieldAlignment, temporalCoupling
    batch_size: int = 32
    training_epochs: int = 1000
    
    # Model checkpoint path
    model_path: str = "wormhole_stability_model.pth"

class WormholeStabilityModel(nn.Module):
    """Neural network model for wormhole stability prediction"""
    def __init__(self, config: WormholeConfig):
        super().__init__()
        self.config = config
        
        self.network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.output_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

@dataclass
class PrimeMediationConfig:
    """Configuration parameters for prime-mediated time modulation"""
    num_primes: int = 100
    time_range: Tuple[float, float] = (1.0, 500.0)
    resolution: int = 1000
    modulation_amplitude: float = 1.0
    frequency_factor: float = 2 * np.pi
    
    # Visualization settings
    plot_width: int = 10
    plot_height: int = 5
    line_color: str = 'b'
    baseline_color: str = 'k'
    baseline_style: str = '--'
    baseline_alpha: float = 0.5
    
    # Add wormhole configuration
    wormhole_config: WormholeConfig = field(default_factory=WormholeConfig)

class PrimeModulator:
    """Handles prime-based time modulation calculations with wormhole stability"""
    def __init__(self, config: Optional[PrimeMediationConfig] = None):
        self.config = config or PrimeMediationConfig()
        self._prime_cache: List[int] = []
        self._initialize_primes()
        
        # Initialize wormhole stability model
        self.wormhole_model = WormholeStabilityModel(self.config.wormhole_config)
        self._load_wormhole_model()
    
    def _initialize_primes(self):
        """Initialize prime number cache"""
        self._prime_cache = [prime(n) for n in range(1, self.config.num_primes + 1)]
        self._prime_array = np.array(self._prime_cache)
    
    def _load_wormhole_model(self):
        """Load pre-trained wormhole model if available"""
        try:
            self.wormhole_model.load_state_dict(
                torch.load(self.config.wormhole_config.model_path)
            )
            self.wormhole_model.eval()
        except FileNotFoundError:
            print("No pre-trained wormhole model found. Using initialized weights.")
    
    def calculate_wormhole_stability(
        self,
        throat_radius: float,
        energy_density: float,
        field_strength: float,
        temporal_flow: float
    ) -> Dict[str, float]:
        """Calculate wormhole stability metrics"""
        with torch.no_grad():
            inputs = torch.tensor([
                throat_radius,
                energy_density,
                field_strength,
                temporal_flow
            ], dtype=torch.float32).unsqueeze(0)
            
            outputs = self.wormhole_model(inputs).squeeze(0)
            
            return {
                'lambda': float(outputs[0]),
                'stability': float(outputs[1]),
                'coherence': float(outputs[2]),
                'wormhole_integrity': float(outputs[3]),
                'field_alignment': float(outputs[4]),
                'temporal_coupling': float(outputs[5])
            }
    
    def prime_modulated_time(self, t: float) -> float:
        """Calculate prime-modulated time value"""
        modulation = np.sum(
            np.sin(self.config.frequency_factor * t / self._prime_array)
        ) / self.config.num_primes
        return modulation * self.config.modulation_amplitude
    
    def generate_time_series(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Generate time series data with wormhole stability metrics"""
        t_values, modulation_values = self.generate_time_series()
        
        # Calculate wormhole stability based on modulation
        avg_modulation = np.mean(np.abs(modulation_values))
        stability_metrics = self.calculate_wormhole_stability(
            throat_radius=avg_modulation,
            energy_density=np.std(modulation_values),
            field_strength=np.max(np.abs(modulation_values)),
            temporal_flow=np.mean(np.diff(modulation_values))
        )
        
        return t_values, modulation_values, stability_metrics
    
    def plot_modulation(self, show: bool = True):
        """Plot the prime modulation function"""
        t_values, modulation_values = self.generate_time_series()
        
        plt.figure(figsize=(self.config.plot_width, self.config.plot_height))
        plt.plot(
            t_values,
            modulation_values,
            label='Prime-Modulated Time Function',
            color=self.config.line_color
        )
        plt.axhline(
            0,
            color=self.config.baseline_color,
            linestyle=self.config.baseline_style,
            alpha=self.config.baseline_alpha
        )
        plt.xlabel("Time")
        plt.ylabel("Modulation Amplitude")
        plt.title("Prime-Mediated Time Modulation")
        plt.legend()
        
        if show:
            plt.show()

def create_default_modulator() -> PrimeModulator:
    """Create a PrimeModulator instance with default configuration"""
    return PrimeModulator()

def train_wormhole_model(config: WormholeConfig = None) -> WormholeStabilityModel:
    """Train a new wormhole stability model"""
    if config is None:
        config = WormholeConfig()
        
    model = WormholeStabilityModel(config)
    trainer = WormholeStabilityTrainer(model, config.learning_rate)
    
    for epoch in range(config.training_epochs):
        avg_loss = trainer.train_epoch(config.batch_size)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Save the trained model
    torch.save(model.state_dict(), config.model_path)
    return model

# Example usage
if __name__ == "__main__":
    # Create modulator with default config
    modulator = create_default_modulator()
    
    # Train wormhole model if needed
    if not os.path.exists(modulator.config.wormhole_config.model_path):
        print("Training new wormhole stability model...")
        train_wormhole_model(modulator.config.wormhole_config)
    
    # Generate and plot modulation with stability metrics
    t_values, modulation_values, stability = modulator.generate_time_series()
    modulator.plot_modulation()
    
    print("\nWormhole Stability Metrics:")
    for metric, value in stability.items():
        print(f"{metric}: {value:.4f}")

# Use default config
model = train_model()

# Or customize training
config = TrainingConfig(
    epochs=2000,
    batch_size=64,
    learning_rate=0.001
)
model = train_model(config) 