import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

@dataclass
class OptimizerConfig:
    """Configuration for Prime Fractal Optimizer"""
    primes: List[int] = None
    low_zeros: List[float] = None
    alpha: float = 0.1
    n_terms: int = 100
    x_init: float = 1.0
    method: str = 'Nelder-Mead'
    max_iter: int = 1000
    
    def __post_init__(self):
        if self.primes is None:
            self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        if self.low_zeros is None:
            self.low_zeros = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351]

class PrimeFractalOptimizer:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.optimization_history = []
        
    def prime_contributions(self, x: float) -> float:
        """Enhanced prime-based oscillatory contributions"""
        return sum(np.cos(2 * np.pi * x / p) for p in self.config.primes)
    
    def fractal_dynamics(self, x: float) -> float:
        """Multi-scale fractal dynamics"""
        return sum(np.log(1 + x / n) for n in range(1, self.config.n_terms + 1))
    
    def zeta_stability(self, t: float) -> float:
        """Riemann zeta zeros stability contribution"""
        return sum(np.cos(gamma * np.log(t)) for gamma in self.config.low_zeros)
    
    def quantum_tunneling(self, x: float) -> float:
        """Quantum-inspired tunneling effect"""
        return np.exp(-self.config.alpha * x)
    
    def total_energy(self, x: float) -> float:
        """Calculate total system energy"""
        prime_contrib = self.prime_contributions(x)
        fractal_contrib = self.fractal_dynamics(x)
        zeta_contrib = self.zeta_stability(x)
        tunneling_contrib = self.quantum_tunneling(x)
        
        energy = prime_contrib + fractal_contrib + zeta_contrib - tunneling_contrib
        self.optimization_history.append((x, energy))
        return energy
    
    def optimize(self) -> Tuple[float, float]:
        """Perform optimization"""
        self.optimization_history = []
        
        result = minimize(
            self.total_energy,
            self.config.x_init,
            method=self.config.method,
            options={'maxiter': self.config.max_iter}
        )
        
        return result.x[0], result.fun
    
    def visualize_optimization(self):
        """Visualize optimization progress"""
        if not self.optimization_history:
            print("No optimization history available. Run optimize() first.")
            return
            
        x_values, energies = zip(*self.optimization_history)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(x_values, 'b-', label='State')
        plt.xlabel('Iteration')
        plt.ylabel('State Value')
        plt.title('State Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(122)
        plt.plot(energies, 'r-', label='Energy')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_landscape(self, x_range: np.ndarray):
        """Analyze energy landscape"""
        energies = [self.total_energy(x) for x in x_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, energies, 'g-', label='Energy Landscape')
        plt.scatter(self.optimization_history[0][0], self.optimization_history[0][1], 
                   color='r', label='Initial Point')
        plt.scatter(self.optimization_history[-1][0], self.optimization_history[-1][1], 
                   color='b', label='Final Point')
        plt.xlabel('State')
        plt.ylabel('Energy')
        plt.title('Energy Landscape Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()

def create_optimizer(
    primes: Optional[List[int]] = None,
    low_zeros: Optional[List[float]] = None,
    alpha: float = 0.1,
    n_terms: int = 100,
    x_init: float = 1.0,
    method: str = 'Nelder-Mead',
    max_iter: int = 1000
) -> PrimeFractalOptimizer:
    """Factory function to create PrimeFractalOptimizer instance"""
    config = OptimizerConfig(
        primes=primes,
        low_zeros=low_zeros,
        alpha=alpha,
        n_terms=n_terms,
        x_init=x_init,
        method=method,
        max_iter=max_iter
    )
    return PrimeFractalOptimizer(config)

# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = create_optimizer()
    
    # Perform optimization
    optimal_x, optimal_energy = optimizer.optimize()
    print(f"Optimal Solution: {optimal_x:.6f}")
    print(f"Energy at Optimal Solution: {optimal_energy:.6f}")
    
    # Visualize optimization progress
    optimizer.visualize_optimization()
    
    # Analyze energy landscape
    x_range = np.linspace(0, 10, 1000)
    optimizer.analyze_landscape(x_range) 