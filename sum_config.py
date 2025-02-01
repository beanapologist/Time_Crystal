"""
Configuration for Sustainable Universal Metrics (SUM) System
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

@dataclass
class SystemConstants:
    """Physical and computational constants"""
    MATRIX_SIZE: int = 4
    REGULARIZATION: float = 1e-6
    DEFAULT_CAPACITY: float = 0.5
    DEFAULT_ENTROPY: float = 0.5
    DEFAULT_COHERENCE: float = 0.5
    
    # Time constants
    SECONDS_PER_DAY: int = 86400
    SIMULATION_INTERVAL: int = 5
    
    # Physics constants
    PLANCK_CONSTANT: float = 6.62607015e-34
    BOLTZMANN_CONSTANT: float = 1.380649e-23
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11

@dataclass
class OptimizationConfig:
    """Configuration for quantum optimization"""
    matrix_size: int = 4
    learning_rate: float = 0.001
    regularization: float = 1e-6
    cache_size: int = 1000
    
    # Quantum parameters
    coherence_threshold: float = 0.8
    entanglement_density: float = 0.5
    quantum_coupling: float = 0.99
    
    # Magnetic field parameters
    field_gradient: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125]
    )

@dataclass
class TransactionConfig:
    """Configuration for transaction processing"""
    min_amount: float = 0.0
    max_amount: float = 1e6
    min_eco_impact: float = 0.0
    max_eco_impact: float = 100.0
    
    # Value computation
    value_scaling: float = 1.0
    impact_weight: float = 0.5
    efficiency_threshold: float = 0.7

@dataclass
class SimulationConfig:
    """Configuration for system simulation"""
    duration_seconds: int = 300
    update_interval: int = 5
    log_interval: int = 10
    
    # Transaction generation
    amount_amplitude: float = 100.0
    amount_frequency: float = 1/60  # Hz
    eco_amplitude: float = 10.0
    eco_frequency: float = 1/120  # Hz

@dataclass
class SUMConfig:
    """Main configuration for SUM system"""
    constants: SystemConstants = field(default_factory=SystemConstants)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    transaction: TransactionConfig = field(default_factory=TransactionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Device configuration
    device: torch.device = field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check matrix size consistency
            assert self.constants.MATRIX_SIZE == self.optimization.matrix_size
            
            # Check value ranges
            assert 0 <= self.optimization.coherence_threshold <= 1
            assert 0 <= self.optimization.entanglement_density <= 1
            assert 0 <= self.optimization.quantum_coupling <= 1
            
            # Check transaction limits
            assert self.transaction.min_amount < self.transaction.max_amount
            assert self.transaction.min_eco_impact < self.transaction.max_eco_impact
            
            # Check simulation parameters
            assert self.simulation.duration_seconds > 0
            assert self.simulation.update_interval > 0
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False

def create_default_config() -> SUMConfig:
    """Create default SUM configuration"""
    return SUMConfig()

def load_config(path: str) -> SUMConfig:
    """Load configuration from file"""
    # TODO: Implement configuration loading from file
    return create_default_config()

def save_config(config: SUMConfig, path: str):
    """Save configuration to file"""
    # TODO: Implement configuration saving to file
    pass

if __name__ == "__main__":
    # Create and validate default configuration
    config = create_default_config()
    if config.validate():
        print("Configuration validation successful")
        
        # Print some key parameters
        print("\nKey Configuration Parameters:")
        print(f"Matrix Size: {config.optimization.matrix_size}")
        print(f"Quantum Coupling: {config.optimization.quantum_coupling}")
        print(f"Device: {config.device}")
        print(f"Simulation Duration: {config.simulation.duration_seconds}s") 