"""
Quantum Clock Implementation for Time Crystal System
"""

import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class QuantumClockConfig:
    """Configuration parameters for quantum clock"""
    base_frequency: float = 1.0  # Base oscillation frequency in Hz
    phase_stability: float = 0.99  # Phase stability factor (0-1)
    coherence_threshold: float = 0.95  # Minimum coherence threshold
    harmonic_factor: float = 1.618  # Golden ratio for harmonic oscillations
    dilation_strength: float = 0.1  # Strength of time dilation effects
    quantum_noise_factor: float = 0.01  # Quantum noise influence
    measurement_interval: float = 0.001  # Time between measurements in seconds

class QuantumClock:
    """
    Quantum Clock that manages time evolution in the quantum system
    with support for time dilation and phase stability measurements
    """
    def __init__(self, config: Optional[QuantumClockConfig] = None):
        self.config = config or QuantumClockConfig()
        self.start_time = time.time()
        self.last_tick = self.start_time
        self.phase = 0.0
        self.coherence = 1.0
        
    def tick(self) -> Dict[str, float]:
        """
        Advance the quantum clock and return time metrics
        
        Returns:
            Dict containing time-related measurements:
            - phase_stability: Measure of phase coherence
            - stability: Overall clock stability
            - dilation: Current time dilation factor
            - coherence: Quantum state coherence
            - harmonic_alignment: Measure of harmonic resonance
        """
        current_time = time.time()
        delta_t = current_time - self.last_tick
        
        # Calculate base phase evolution
        self.phase += (2 * np.pi * self.config.base_frequency * delta_t)
        self.phase %= (2 * np.pi)
        
        # Apply quantum noise effects
        noise = np.random.normal(0, self.config.quantum_noise_factor)
        
        # Calculate phase stability with noise influence
        phase_stability = self._calculate_phase_stability(noise)
        
        # Calculate time dilation factor
        dilation = self._calculate_time_dilation(delta_t)
        
        # Update coherence
        self.coherence = self._update_coherence(phase_stability)
        
        # Calculate harmonic alignment
        harmonic_alignment = self._calculate_harmonic_alignment()
        
        # Update last tick time
        self.last_tick = current_time
        
        return {
            'phase_stability': phase_stability,
            'stability': self.coherence,
            'dilation': dilation,
            'coherence': self.coherence,
            'harmonic_alignment': harmonic_alignment
        }
    
    def _calculate_phase_stability(self, noise: float) -> float:
        """Calculate phase stability with quantum effects"""
        stability = self.config.phase_stability * (1.0 - abs(noise))
        return max(0.0, min(1.0, stability))
    
    def _calculate_time_dilation(self, delta_t: float) -> float:
        """Calculate time dilation factor"""
        # Use harmonic oscillation for dilation
        oscillation = math.sin(self.phase * self.config.harmonic_factor)
        dilation = 1.0 + (oscillation * self.config.dilation_strength)
        return dilation
    
    def _update_coherence(self, phase_stability: float) -> float:
        """Update quantum coherence based on phase stability"""
        target_coherence = phase_stability * self.config.coherence_threshold
        # Smooth coherence transition
        self.coherence = (0.95 * self.coherence + 0.05 * target_coherence)
        return max(0.0, min(1.0, self.coherence))
    
    def _calculate_harmonic_alignment(self) -> float:
        """Calculate harmonic alignment factor"""
        # Use golden ratio harmonics
        harmonic = math.sin(self.phase * self.config.harmonic_factor)
        alignment = (harmonic + 1) / 2  # Normalize to [0,1]
        return alignment
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since clock initialization"""
        return time.time() - self.start_time
    
    def reset(self):
        """Reset the quantum clock to initial state"""
        self.start_time = time.time()
        self.last_tick = self.start_time
        self.phase = 0.0
        self.coherence = 1.0 