"""
Configuration settings for quantum system components
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class UniversalConstants:
    """Physical constants used in quantum calculations"""
    PLANCK_LENGTH: float = 1.616255e-35
    PLANCK_MASS: float = 2.176434e-8
    PLANCK_TIME: float = 5.391247e-44
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11
    SPEED_OF_LIGHT: float = 299792458.0
    PHI: float = 1.618034  # Golden ratio

@dataclass
class QuantumSystemConfig:
    """Main configuration for quantum system"""
    # Core parameters
    coherence_threshold: float = 0.7
    max_validators: int = 100
    stability_threshold: float = 0.95
    quantum_depth: int = 4
    entanglement_density: float = 0.5
    quantum_efficiency: float = 0.99
    reality_integrity: float = 0.9
    critical_coupling: float = 0.8
    lambda_critical: float = 0.5
    
    # Model dimensions
    input_dim: int = 512
    hidden_dim: int = 256
    output_dim: int = 128
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 100
    
    # System thresholds
    QUANTUM_EFFICIENCY: float = 0.99
    STABILITY_THRESHOLD: float = 0.95

    def __post_init__(self):
        self.constants = UniversalConstants()

def create_default_config() -> QuantumSystemConfig:
    """Create default configuration instance"""
    return QuantumSystemConfig() 