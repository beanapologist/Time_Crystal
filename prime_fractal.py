import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.optimize import minimize

@dataclass
class PrimeFractalConfig:
    """Configuration for Prime Fractal system"""
    primes: List[int] = None
    scales: List[float] = None
    n_terms: int = 100
    resonance_frequency: float = 10.0
    amplitude: float = 0.1
    phase_shift: float = 0.0
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.primes is None:
            self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        if self.scales is None:
            self.scales = [1.0, 10.0, 100.0]

class PrimeFractal:
    def __init__(self, config: PrimeFractalConfig):
        self.config = config
        self.state = None
        self.energy = None
        
    def enhanced_prime_contributions(self, x: float, phase: float = 0) -> float:
        """Calculate prime-based oscillatory contributions with phase shifts"""
        return sum(np.cos(2 * np.pi * x / p + phase) for p in self.config.primes)
    
    def multi_scale_fractal_dynamics(self, x: float) -> float:
        """Implement fractal patterns at multiple scales"""
        return sum(
            sum(np.log(1 + x / (n * scale)) 
                for n in range(1, self.config.n_terms + 1))
            for scale in self.config.scales
        )
    
    def quantum_resonance(self, t: float) -> float:
        """Calculate quantum resonance effects"""
        freq = self.config.resonance_frequency
        amp = self.config.amplitude
        return amp * np.sin(2 * np.pi * t * freq + self.config.phase_shift)
    
    def total_energy(self, x: float, t: float) -> float:
        """Calculate total system energy"""
        prime_contrib = self.enhanced_prime_contributions(x, self.config.phase_shift)
        fractal_contrib = self.multi_scale_fractal_dynamics(x)
        resonance = self.quantum_resonance(t)
        return prime_contrib + fractal_contrib + resonance
    
    def optimize_state(self, t: float, x_init: float = 1.0) -> Tuple[float, float]:
        """Optimize system state for given time"""
        result = minimize(
            lambda x: -self.total_energy(x[0], t),  # Negative for maximization
            x0=[x_init],
            method='Nelder-Mead'
        )
        return result.x[0], -result.fun
    
    def evolve(self, t_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve system over time range"""
        states = []
        energies = []
        current_state = 1.0
        
        for t in t_range:
            current_state, energy = self.optimize_state(t, current_state)
            states.append(current_state)
            energies.append(energy)
            
        return np.array(states), np.array(energies)
    
    def visualize(self, t_range: np.ndarray, states: np.ndarray, energies: np.ndarray):
        """Visualize system evolution"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot states
        ax1.plot(t_range, states, 'b-', label='System State')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('State')
        ax1.set_title('Prime Fractal State Evolution')
        ax1.legend()
        ax1.grid(True)
        
        # Plot energies
        ax2.plot(t_range, energies, 'r-', label='System Energy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy')
        ax2.set_title('Prime Fractal Energy Evolution')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def create_prime_fractal(
    primes: Optional[List[int]] = None,
    scales: Optional[List[float]] = None,
    n_terms: int = 100,
    resonance_frequency: float = 10.0,
    amplitude: float = 0.1,
    phase_shift: float = 0.0,
    learning_rate: float = 0.001
) -> PrimeFractal:
    """Factory function to create PrimeFractal instance"""
    config = PrimeFractalConfig(
        primes=primes,
        scales=scales,
        n_terms=n_terms,
        resonance_frequency=resonance_frequency,
        amplitude=amplitude,
        phase_shift=phase_shift,
        learning_rate=learning_rate
    )
    return PrimeFractal(config)

# Example usage
if __name__ == "__main__":
    # Create prime fractal system
    fractal = create_prime_fractal()
    
    # Generate time range
    t_range = np.linspace(0, 10, 1000)
    
    # Evolve system
    states, energies = fractal.evolve(t_range)
    
    # Visualize results
    fractal.visualize(t_range, states, energies) 