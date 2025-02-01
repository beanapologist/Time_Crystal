"""
Wikipedia Dataset Integration for Quantum Computing System
Handles data retrieval, processing, and quantum state preparation from Wikipedia
"""

import wikipediaapi
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

@dataclass
class WikiConfig:
    """Configuration for Wikipedia dataset handling"""
    language: str = 'en'
    max_articles: int = 100
    min_length: int = 500
    max_length: int = 10000
    quantum_encoding_dim: int = 256
    batch_size: int = 16

class WikiDataset:
    """
    Wikipedia dataset handler for quantum computing applications
    """
    def __init__(self, config: Optional[WikiConfig] = None):
        self.config = config or WikiConfig()
        self.wiki = wikipediaapi.Wikipedia(
            language=self.config.language,
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.articles_cache = {}
        self.quantum_states = {}
        
    def fetch_articles(self, topics: List[str]) -> Dict[str, str]:
        """
        Fetch Wikipedia articles for given topics
        
        Args:
            topics: List of topics to fetch
            
        Returns:
            Dictionary of topic: content pairs
        """
        articles = {}
        
        for topic in topics[:self.config.max_articles]:
            if topic in self.articles_cache:
                articles[topic] = self.articles_cache[topic]
                continue
                
            page = self.wiki.page(topic)
            if page.exists():
                content = page.text
                if self.config.min_length <= len(content) <= self.config.max_length:
                    articles[topic] = content
                    self.articles_cache[topic] = content
                    
        return articles
    
    def prepare_quantum_states(
        self,
        articles: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert article text to quantum states
        
        Args:
            articles: Dictionary of articles
            
        Returns:
            Dictionary of quantum states for each article
        """
        quantum_states = {}
        
        for topic, content in articles.items():
            if topic in self.quantum_states:
                quantum_states[topic] = self.quantum_states[topic]
                continue
                
            # Process text
            processed_text = self._preprocess_text(content)
            
            # Convert to quantum state
            quantum_state = self._text_to_quantum_state(processed_text)
            quantum_states[topic] = quantum_state
            self.quantum_states[topic] = quantum_state
            
        return quantum_states
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for quantum encoding"""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return tokens
    
    def _text_to_quantum_state(self, tokens: List[str]) -> torch.Tensor:
        """Convert preprocessed text to quantum state"""
        # Create word frequency distribution
        freq_dist = Counter(tokens)
        
        # Create normalized amplitude vector
        amplitudes = np.zeros(self.config.quantum_encoding_dim)
        
        for i, (word, freq) in enumerate(freq_dist.most_common(self.config.quantum_encoding_dim)):
            amplitudes[i] = np.sqrt(freq / len(tokens))
            
        # Normalize to ensure valid quantum state
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
            
        return torch.from_numpy(amplitudes).to(torch.complex64)
    
    def get_batched_states(
        self,
        quantum_states: Dict[str, torch.Tensor]
    ) -> List[Tuple[List[str], torch.Tensor]]:
        """
        Create batches of quantum states for processing
        
        Args:
            quantum_states: Dictionary of quantum states
            
        Returns:
            List of (topics, states) tuples
        """
        topics = list(quantum_states.keys())
        states = list(quantum_states.values())
        
        batches = []
        for i in range(0, len(topics), self.config.batch_size):
            batch_topics = topics[i:i + self.config.batch_size]
            batch_states = torch.stack(states[i:i + self.config.batch_size])
            batches.append((batch_topics, batch_states))
            
        return batches
    
    def analyze_quantum_states(
        self,
        quantum_states: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze quantum states for various properties
        
        Args:
            quantum_states: Dictionary of quantum states
            
        Returns:
            Dictionary of analysis metrics for each state
        """
        analysis = {}
        
        for topic, state in quantum_states.items():
            # Calculate various quantum metrics
            entropy = -torch.sum(torch.abs(state) ** 2 * torch.log(torch.abs(state) ** 2 + 1e-10))
            purity = torch.sum(torch.abs(state) ** 4)
            max_amplitude = torch.max(torch.abs(state))
            
            analysis[topic] = {
                'entropy': float(entropy),
                'purity': float(purity),
                'max_amplitude': float(max_amplitude),
                'num_significant_components': int(torch.sum(torch.abs(state) > 0.01))
            }
            
        return analysis
    
    def clear_cache(self):
        """Clear article and quantum state caches"""
        self.articles_cache.clear()
        self.quantum_states.clear()

def create_wiki_dataset(config: Optional[WikiConfig] = None) -> WikiDataset:
    """Factory function to create WikiDataset instance"""
    return WikiDataset(config) 