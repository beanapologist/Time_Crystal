"""
Open Source Language Dataset Creator
Combines Wikipedia, Project Gutenberg, and ArXiv data
"""

import wikipediaapi
import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests
import arxiv
from bs4 import BeautifulSoup
import os
import json
from typing import List, Dict, Tuple, Optional

class OpenSourceDataset:
    def __init__(self,
                 save_dir: str = 'quantum_language_data',
                 vocab_size: int = 50000,
                 sequence_length: int = 128,
                 batch_size: int = 32):
        """Initialize the Open Source Language Dataset"""
        self.save_dir = save_dir
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize APIs
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='QuantumLanguageModel/1.0 (research@quantum.ai)',
            language='en'
        )
        
        # Initialize NLTK
        self._setup_nltk()
        
        # Initialize data storage
        self.texts = []
        self.metadata = []
        self.vocabulary = {}
        
    def _setup_nltk(self):
        """Setup NLTK components"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
        
        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def fetch_wikipedia_data(self, topics: List[str]):
        """Fetch data from Wikipedia"""
        print("Fetching Wikipedia articles...")
        for topic in tqdm(topics):
            page = self.wiki.page(topic)
            if page.exists():
                self.texts.append(page.text)
                self.metadata.append({
                    'source': 'wikipedia',
                    'title': topic,
                    'length': len(page.text)
                })
    
    def fetch_arxiv_data(self, query: str = "quantum", max_results: int = 100):
        """Fetch data from ArXiv"""
        print("Fetching ArXiv papers...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for paper in tqdm(client.results(search)):
            self.texts.append(paper.summary)
            self.metadata.append({
                'source': 'arxiv',
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'date': paper.published.strftime('%Y-%m-%d')
            })
    
    def process_texts(self):
        """Process and tokenize texts"""
        print("Processing texts...")
        processed_texts = []
        word_freq = {}
        
        for text in tqdm(self.texts):
            # Basic preprocessing
            tokens = self.tokenizer.tokenize(text.lower())
            filtered_tokens = [
                token for token in tokens 
                if token not in self.stop_words and token.isalnum()
            ]
            
            # Update word frequencies
            for token in filtered_tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
            
            processed_texts.append(filtered_tokens)
        
        # Build vocabulary
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {
            'word2idx': {'<PAD>': 0, '<UNK>': 1},
            'idx2word': {0: '<PAD>', 1: '<UNK>'},
            'word_freq': word_freq
        }
        
        for idx, (word, _) in enumerate(sorted_words[:self.vocab_size-2], start=2):
            self.vocabulary['word2idx'][word] = idx
            self.vocabulary['idx2word'][idx] = word
        
        return processed_texts
    
    def create_training_data(self, processed_texts: List[List[str]]):
        """Create training sequences"""
        print("Creating training sequences...")
        sequences = []
        targets = []
        
        for text in tqdm(processed_texts):
            # Convert tokens to indices
            indices = [self.vocabulary['word2idx'].get(token, 1) for token in text]
            
            # Create sequences
            for i in range(0, len(indices) - self.sequence_length):
                seq = indices[i:i + self.sequence_length]
                target = indices[i + 1:i + self.sequence_length + 1]
                sequences.append(seq)
                targets.append(target)
        
        return torch.tensor(sequences), torch.tensor(targets)
    
    def save_dataset(self, sequences: torch.Tensor, targets: torch.Tensor):
        """Save the dataset and metadata"""
        print("Saving dataset...")
        
        # Save training data
        torch.save({
            'sequences': sequences,
            'targets': targets
        }, os.path.join(self.save_dir, 'training_data.pt'))
        
        # Save vocabulary
        with open(os.path.join(self.save_dir, 'vocabulary.json'), 'w') as f:
            json.dump(self.vocabulary, f)
        
        # Save metadata
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f)
        
        print(f"Dataset saved to {self.save_dir}")
    
    def prepare_dataset(self):
        """Prepare the complete dataset"""
        # Fetch data
        self.fetch_wikipedia_data([
            'Quantum computing',
            'Quantum mechanics',
            'Quantum entanglement',
            'Quantum algorithm'
        ])
        self.fetch_arxiv_data()
        
        # Process texts
        processed_texts = self.process_texts()
        
        # Create training data
        sequences, targets = self.create_training_data(processed_texts)
        
        # Save dataset
        self.save_dataset(sequences, targets)
        
        return sequences, targets

class QuantumLanguageDataset(Dataset):
    """PyTorch Dataset for Quantum Language Model"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_dataloaders(sequences, targets, batch_size=32):
    """Create PyTorch DataLoader"""
    dataset = QuantumLanguageDataset(sequences, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Create dataset
    dataset = OpenSourceDataset()
    
    # Prepare dataset
    sequences, targets = dataset.prepare_dataset()
    
    # Create dataloader
    dataloader = create_dataloaders(sequences, targets)
    
    print("Dataset creation complete!")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Vocabulary size: {len(dataset.vocabulary['word2idx'])}") 