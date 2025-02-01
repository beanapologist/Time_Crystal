"""
Quantum Language Dataset Implementation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class QuantumLanguageDataset(Dataset):
    def __init__(self,
                 sequence_length: int = 512,
                 batch_size: int = 32,
                 tokenizer_name: str = 'bert-base-scientific-cased'):
        """Initialize Quantum Language Dataset"""
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize storage
        self.texts = []
        self.encodings = []
        self.metadata = []
    
    def add_text(self, text: str, metadata: Optional[Dict] = None):
        """Add text to dataset"""
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def process_texts(self):
        """Process all texts using tokenizer"""
        for text in tqdm(self.texts, desc="Processing texts"):
            encoding = self.tokenizer(
                text,
                max_length=self.sequence_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.encodings.append(encoding['input_ids'].squeeze())
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]
    
    def get_metadata(self, idx):
        """Get metadata for a specific item"""
        return self.metadata[idx]
    
    def save_state(self, path: str):
        """Save dataset state"""
        torch.save({
            'texts': self.texts,
            'encodings': self.encodings,
            'metadata': self.metadata
        }, path)
    
    def load_state(self, path: str):
        """Load dataset state"""
        state = torch.load(path)
        self.texts = state['texts']
        self.encodings = state['encodings']
        self.metadata = state['metadata']

def create_dataloaders(dataset: QuantumLanguageDataset,
                      batch_size: Optional[int] = None,
                      train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    if batch_size is None:
        batch_size = dataset.batch_size
    
    # Calculate split sizes
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    dataset = QuantumLanguageDataset()
    
    # Add some example texts
    texts = [
        "Quantum computing uses quantum phenomena",
        "Superposition is a key quantum principle"
    ]
    
    for text in texts:
        dataset.add_text(text)
    
    # Process texts
    dataset.process_texts()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dataset)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}") 