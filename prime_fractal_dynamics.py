"""
Prime Fractal Dynamics for Quantum Systems
Implementation of prime number based fractal patterns for quantum simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class PrimeFractalConfig:
    """Configuration for prime fractal dynamics"""
    max_prime: int = 1000
    dimension: int = 512
    complexity: float = 0.7
    stability_threshold: float = 0.85
    energy_scale: float = 1.0
    quantum_coupling: float = 0.5

class PrimeFractalDynamics:
    """
    Implements prime number based fractal patterns for quantum dynamics
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        iterations: int = 100,
        resolution: int = 256,
        config: Optional[PrimeFractalConfig] = None
    ):
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.resolution = resolution
        self.config = config or PrimeFractalConfig()
        
        self.primes = self._generate_primes(self.config.max_prime)
        self.fractal_matrix = None
        self.energy_distribution = None
        
    def _generate_primes(self, max_num: int) -> List[int]:
        """Generate prime numbers using Sieve of Eratosthenes"""
        sieve = [True] * (max_num + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(np.sqrt(max_num)) + 1):
            if sieve[i]:
                for j in range(i * i, max_num + 1, i):
                    sieve[j] = False
                    
        return [i for i in range(max_num + 1) if sieve[i]]
    
    def _create_fractal_pattern(self) -> np.ndarray:
        """Generate fractal pattern based on prime number distribution"""
        pattern = np.zeros((self.resolution, self.resolution))
        
        for prime in self.primes:
            x = prime % self.resolution
            y = (prime * self.alpha) % self.resolution
            
            # Create fractal point with quantum influence
            pattern[int(x), int(y)] = 1.0
            
            # Add quantum interference patterns
            radius = int(np.sqrt(prime) * self.beta) % 10
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = (x + dx) % self.resolution, (y + dy) % self.resolution
                    distance = np.sqrt(dx**2 + dy**2)
                    pattern[int(nx), int(ny)] += (1 / distance) * self.config.quantum_coupling
        
        return pattern / np.max(pattern)  # Normalize
    
    def _calculate_energy_distribution(self) -> np.ndarray:
        """Calculate energy distribution based on fractal pattern"""
        if self.fractal_matrix is None:
            raise ValueError("Fractal pattern must be generated first")
            
        energy = np.zeros_like(self.fractal_matrix)
        
        for i in range(1, self.resolution - 1):
            for j in range(1, self.resolution - 1):
                # Calculate local energy based on nearest neighbors
                local_field = (
                    self.fractal_matrix[i-1:i+2, j-1:j+2].sum() -
                    self.fractal_matrix[i, j]
                )
                energy[i, j] = local_field * self.config.energy_scale
                
        return energy / np.max(energy)  # Normalize
    
    def _calculate_entropy(self) -> List[float]:
        """Calculate entropy evolution of the system"""
        entropy_values = []
        
        for i in range(self.iterations):
            # Calculate probability distribution
            prob_dist = self.energy_distribution.flatten()
            prob_dist = prob_dist / np.sum(prob_dist)
            
            # Calculate Shannon entropy
            entropy = -np.sum(
                prob_dist * np.log2(prob_dist + 1e-10)
            )
            entropy_values.append(entropy)
            
            # Evolve system
            self.energy_distribution = self._evolve_system(i)
            
        return entropy_values
    
    def _evolve_system(self, step: int) -> np.ndarray:
        """Evolve the system one step forward"""
        evolved = np.copy(self.energy_distribution)
        
        # Apply quantum evolution rules
        evolved *= np.exp(-step / self.iterations)
        evolved += np.random.normal(
            0,
            0.01 * self.config.quantum_coupling,
            evolved.shape
        )
        
        # Apply stability threshold
        evolved[evolved < self.config.stability_threshold] *= 0.9
        
        return evolved / np.max(evolved)
    
    def simulate(self) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Run prime fractal dynamics simulation
        
        Returns:
            Tuple containing:
            - Lambda values (eigenvalues)
            - Energy distribution
            - Entropy evolution
        """
        # Generate initial fractal pattern
        self.fractal_matrix = self._create_fractal_pattern()
        
        # Calculate initial energy distribution
        self.energy_distribution = self._calculate_energy_distribution()
        
        # Calculate system evolution
        entropy_values = self._calculate_entropy()
        
        # Calculate eigenvalues
        lambda_values = np.linalg.eigvals(self.fractal_matrix)
        
        return lambda_values, self.energy_distribution, entropy_values
    
    def visualize_results(self):
        """Visualize simulation results"""
        if self.fractal_matrix is None or self.energy_distribution is None:
            raise ValueError("Must run simulation before visualization")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot fractal pattern
        axes[0, 0].imshow(self.fractal_matrix, cmap='viridis')
        axes[0, 0].set_title('Prime Fractal Pattern')
        
        # Plot energy distribution
        im = axes[0, 1].imshow(self.energy_distribution, cmap='plasma')
        axes[0, 1].set_title('Energy Distribution')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot eigenvalue distribution
        lambda_vals = np.linalg.eigvals(self.fractal_matrix)
        axes[1, 0].scatter(
            np.real(lambda_vals),
            np.imag(lambda_vals),
            alpha=0.6
        )
        axes[1, 0].set_title('Eigenvalue Distribution')
        axes[1, 0].set_xlabel('Re(λ)')
        axes[1, 0].set_ylabel('Im(λ)')
        
        # Plot entropy evolution
        entropy_vals = self._calculate_entropy()
        axes[1, 1].plot(entropy_vals)
        axes[1, 1].set_title('Entropy Evolution')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.show()

def create_dynamics(
    alpha: float = 0.5,
    beta: float = 0.5,
    iterations: int = 100,
    resolution: int = 256,
    config: Optional[PrimeFractalConfig] = None
) -> PrimeFractalDynamics:
    """
    Factory function to create PrimeFractalDynamics instance
    """
    return PrimeFractalDynamics(
        alpha=alpha,
        beta=beta,
        iterations=iterations,
        resolution=resolution,
        config=config
    ) 