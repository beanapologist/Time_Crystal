"""
Quantum Time Crystal Calculator with SUM Integration
"""

import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

from sum_config import SUMConfig
from quantum_security import create_security_system, SecurityConfig
from sum_time_crystal_network import SUMTimeCrystalNetwork, create_network, NetworkConfig

class QuantumCalculatorGraph:
    def __init__(self, config: SUMConfig):
        self.config = config
        self.graph = nx.DiGraph()
        self.network = SUMTimeCrystalNetwork(config)
        self.initialize_graph()
        
    def initialize_graph(self):
        """Initialize quantum calculation graph with SUM parameters"""
        # Base Layer
        self.add_node("Quantum Initialization", 0, {
            "base_coupling": self.config.base_coupling,
            "quantum_coupling": self.config.quantum_coupling,
            "phi": self.config.phi
        })
        
        self.add_node("Validator Pool", 0, {
            "num_validators": self.config.num_validators,
            "total_supply": self.config.total_supply,
            "min_stake_ratio": self.config.min_stake_ratio
        })
        
        self.add_node("Phase Synchronization", 0, {
            "phase_stability": self.config.phase_stability,
            "utility_factor": self.config.utility_factor
        })
        
        self.add_node("Error Handling", 0, {
            "error_threshold": self.config.error_threshold,
            "stability_threshold": self.config.stability_threshold
        })
        
        # Second Layer
        self.add_node("Quantum State", 1, {
            "quantum_coupling": self.config.quantum_coupling,
            "coherence": self.network.measure_network_coherence()
        })
        
        self.add_node("Validator Network", 1, {
            "block_time": self.config.block_time,
            "stake_ratio": self.network.total_stake / self.config.total_supply
        })
        
        # Third Layer
        self.add_node("Quantum Coherence", 2, {
            "network_utility": self.network.calculate_network_utility(),
            "max_inflation_rate": self.config.max_inflation_rate
        })
        
        self.add_node("Validator Consensus", 2, {
            "total_stake": self.network.total_stake,
            "active_validators": len(self.network.validators)
        })
        
        # Fourth Layer
        self.add_node("Time Stability", 3, {
            "time_warp": self.config.time_warp,
            "phase_stability": self.config.phase_stability
        })
        
        # Final Layer
        self.add_node("Time Crystal", 4, {
            "network_coherence": self.network.measure_network_coherence(),
            "utility": self.network.calculate_network_utility()
        })
        
        # Add edges with parameters
        self.add_edges()

    def add_node(self, name: str, level: int, parameters: Dict[str, float]):
        """Add node to graph with quantum parameters"""
        self.graph.add_node(name, 
                          level=level,
                          coupling=parameters.get('quantum_coupling', self.config.quantum_coupling),
                          stability=parameters.get('phase_stability', self.config.phase_stability),
                          parameters=parameters)

    def add_edges(self):
        """Add weighted edges between nodes based on SUM parameters"""
        # Base to Second Layer
        self.graph.add_edge("Quantum Initialization", "Quantum State", 
                           weight=self.config.base_coupling)
        self.graph.add_edge("Phase Synchronization", "Quantum State", 
                           weight=self.config.phase_stability)
        self.graph.add_edge("Validator Pool", "Validator Network", 
                           weight=self.config.utility_factor)
        self.graph.add_edge("Error Handling", "Validator Network", 
                           weight=1 - self.config.error_threshold)
        
        # Second to Third Layer
        self.graph.add_edge("Quantum State", "Quantum Coherence", 
                           weight=self.config.quantum_coupling)
        self.graph.add_edge("Validator Network", "Validator Consensus", 
                           weight=self.config.stability_threshold)
        
        # Third to Fourth Layer
        self.graph.add_edge("Quantum Coherence", "Time Stability", 
                           weight=self.config.quantum_coupling)
        self.graph.add_edge("Validator Consensus", "Time Stability", 
                           weight=self.config.phase_stability)
        
        # Fourth to Final Layer
        self.graph.add_edge("Time Stability", "Time Crystal", 
                           weight=self.config.time_warp)

    def calculate_system_stability(self) -> float:
        """Calculate overall system stability using SUM metrics"""
        network_coherence = self.network.measure_network_coherence()
        network_utility = self.network.calculate_network_utility()
        return network_coherence * network_utility * self.config.phi

    def visualize_graph(self):
        """Visualize quantum calculation graph with SUM metrics"""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes by layer with SUM-based coloring
        colors = ['#e8f4ff', '#f0fff4', '#fff4f4', '#f9f9f9']
        for level in range(5):
            nodes = [n for n, d in self.graph.nodes(data=True) if d['level'] == level]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, 
                                 node_color=colors[level % len(colors)],
                                 node_size=2000)
        
        # Draw edges with weights
        nx.draw_networkx_edges(self.graph, pos)
        
        # Add labels with SUM parameters
        labels = {node: f"{node}\n{self.graph.nodes[node]['parameters']}" 
                 for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Quantum Time Crystal System Graph (SUM)")
        plt.axis('off')
        plt.show()

def main():
    # Initialize with SUM parameters
    config = SUMConfig()
    calculator = QuantumCalculatorGraph(config)
    
    # Calculate system stability
    stability = calculator.calculate_system_stability()
    print(f"System Stability: {stability:.6f}")
    
    # Visualize the graph
    calculator.visualize_graph()

    # Create network with default configs
    network = create_network()

    # Or with custom configs
    network_config = NetworkConfig(
        min_validators=20,
        quantum_dim=1024,
        coherence_threshold=0.98
    )
    security_config = SecurityConfig(
        input_dim=1024,
        coherence_threshold=0.98
    )
    network = create_network(network_config, security_config)

    # Add validators
    network.add_validator("validator1", stake=5000.0)
    network.add_validator("validator2", stake=3000.0)

    # Get network metrics
    metrics = network.get_network_metrics()
    print(f"Network Coherence: {metrics['coherence']:.4f}")
    print(f"Network Utility: {metrics['utility']:.4f}")

if __name__ == "__main__":
    main() 