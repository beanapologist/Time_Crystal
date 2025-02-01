"""
Fixed Quantum Data Toolkit (QDT) with correct dataset initialization
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm

from language_dataset import QuantumLanguageDataset

@dataclass
class QDTConstants:
    """Constants for Quantum Data Toolkit"""
    # Model Parameters
    HIDDEN_SIZE: int = 768
    NUM_ATTENTION_HEADS: int = 12
    NUM_HIDDEN_LAYERS: int = 6
    DROPOUT_RATE: float = 0.1
    
    # Training Parameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 10
    
    # Data Parameters
    SEQUENCE_LENGTH: int = 512  # Changed from MAX_SEQ_LENGTH
    VOCAB_SIZE: int = 50000
    NUM_METRICS: int = 8
    TRAIN_SPLIT: float = 0.8
    
    # Paths
    BASE_DIR: Path = Path('quantum_system')
    DATA_DIR: Path = Path('quantum_data')
    MODEL_DIR: Path = Path('quantum_models')
    CACHE_DIR: Path = Path('quantum_cache')
    LOG_DIR: Path = Path('quantum_logs')

class QuantumDataToolkit:
    def __init__(self, 
                 constants: Optional[QDTConstants] = None,
                 verbose: bool = True):
        """Initialize Quantum Data Toolkit"""
        self.constants = constants or QDTConstants()
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self._setup_environment()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_environment(self):
        """Setup environment"""
        for dir_path in [self.constants.BASE_DIR, 
                        self.constants.DATA_DIR,
                        self.constants.MODEL_DIR,
                        self.constants.CACHE_DIR,
                        self.constants.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            filename=self.constants.LOG_DIR / 'qdt.log',
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """Initialize components with correct parameters"""
        try:
            # Initialize language dataset with correct parameters
            self.language_dataset = QuantumLanguageDataset(
                sequence_length=self.constants.SEQUENCE_LENGTH,  # Changed parameter name
                batch_size=self.constants.BATCH_SIZE
            )
            logging.info("Language dataset initialized successfully")
            
            # Initialize metrics dataset
            self.metrics_dataset = QuantumMetricsDataset(
                sequence_length=self.constants.NUM_METRICS
            )
            logging.info("Metrics dataset initialized successfully")
            
            # Initialize integration model
            self.integration_model = QuantumLanguageIntegration(
                hidden_size=self.constants.HIDDEN_SIZE,
                num_metrics=self.constants.NUM_METRICS,
                dropout=self.constants.DROPOUT_RATE
            ).to(self.device)
            logging.info("Integration model initialized successfully")
            
        except Exception as e:
            logging.error(f"Component initialization failed: {str(e)}")
            raise
    
    def prepare_datasets(self, 
                        texts: List[str],
                        metrics: torch.Tensor) -> Tuple[torch.utils.data.DataLoader, 
                                                      torch.utils.data.DataLoader]:
        """Prepare datasets"""
        try:
            # Process language data
            for text in tqdm(texts, desc="Processing texts"):
                self.language_dataset.add_text(text)
            self.language_dataset.process_texts()
            logging.info("Language data processed successfully")
            
            # Process metrics data
            for metric in tqdm(metrics, desc="Processing metrics"):
                self.metrics_dataset.add_metric_sequence(
                    metric.numpy(),
                    "quantum_metric"
                )
            logging.info("Metrics data processed successfully")
            
            # Create dataloaders
            train_loader, val_loader = create_integrated_dataloaders(
                texts=texts,
                metrics=metrics,
                batch_size=self.constants.BATCH_SIZE,
                train_split=self.constants.TRAIN_SPLIT
            )
            logging.info("Dataloaders created successfully")
            
            return train_loader, val_loader
            
        except Exception as e:
            logging.error(f"Dataset preparation failed: {str(e)}")
            raise
    
    def save_state(self, path: Optional[Union[str, Path]] = None):
        """Save toolkit state"""
        if path is None:
            path = self.constants.MODEL_DIR / 'qdt_state.pt'
        
        try:
            state_dict = {
                'model_state': self.integration_model.state_dict(),
                'constants': self.constants
            }
            torch.save(state_dict, path)
            logging.info(f"State saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save state: {str(e)}")
            raise
    
    def load_state(self, path: Optional[Union[str, Path]] = None):
        """Load toolkit state"""
        if path is None:
            path = self.constants.MODEL_DIR / 'qdt_state.pt'
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.integration_model.load_state_dict(state_dict['model_state'])
            self.constants = state_dict['constants']
            logging.info(f"State loaded from {path}")
        except Exception as e:
            logging.error(f"Failed to load state: {str(e)}")
            raise

def create_qdt(config: Optional[Dict] = None, verbose: bool = True) -> QuantumDataToolkit:
    """Create QDT instance"""
    try:
        if config:
            constants = QDTConstants(**config)
        else:
            constants = QDTConstants()
        
        return QuantumDataToolkit(constants, verbose)
    except Exception as e:
        logging.error(f"Failed to create QDT: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Create QDT instance
        qdt = create_qdt()
        
        # Example data
        texts = ["Quantum state example", "Another quantum text"]
        metrics = torch.randn(2, 8)  # 2 samples, 8 metrics each
        
        # Prepare datasets
        train_loader, val_loader = qdt.prepare_datasets(texts, metrics)
        
        # Save state
        qdt.save_state()
        
        print("QDT setup completed successfully")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise 