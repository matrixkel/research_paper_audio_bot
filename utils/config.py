"""
Configuration management for the Multi-Agent Research Paper Analysis System.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """
    Configuration class that manages all system settings and API keys.
    
    Handles environment variables and provides default values for system configuration.
    """
    
    # API Configuration
    GROQ_API_KEY: Optional[str] = None
    
    # File Storage Configuration
    STORAGE_BASE_DIR: str = "storage"
    PAPERS_DIR: str = "storage/papers"
    AUDIO_DIR: str = "storage/audio"
    
    # Processing Configuration
    MAX_CONCURRENT_REQUESTS: int = 5
    MAX_PAPERS_PER_SEARCH: int = 50
    DEFAULT_SEARCH_LIMIT: int = 10
    
    # Text Processing Configuration
    MAX_TEXT_LENGTH: int = 10000  # Maximum text length for processing
    SUMMARY_MAX_TOKENS: int = 1000
    SYNTHESIS_MAX_TOKENS: int = 1500
    
    # Audio Configuration
    AUDIO_CLEANUP_DAYS: int = 7  # Days to keep audio files
    TTS_RATE: int = 180  # Words per minute for TTS
    
    # Classification Configuration
    CLASSIFICATION_THRESHOLD: float = 0.3  # Minimum confidence for topic classification
    
    # API Rate Limiting
    GROQ_RATE_LIMIT_DELAY: float = 0.5  # Seconds between API calls
    SEMANTIC_SCHOLAR_RATE_LIMIT: float = 1.0
    ARXIV_RATE_LIMIT: float = 0.5
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    def __init__(self):
        """Initialize configuration with environment variables"""
        self.load_from_environment()
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        # API Keys
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        # Storage Configuration
        self.STORAGE_BASE_DIR = os.getenv("STORAGE_BASE_DIR", "storage")
        self.PAPERS_DIR = os.getenv("PAPERS_DIR", "storage/papers")
        self.AUDIO_DIR = os.getenv("AUDIO_DIR", "storage/audio")
        
        # Processing Configuration
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
        self.MAX_PAPERS_PER_SEARCH = int(os.getenv("MAX_PAPERS_PER_SEARCH", "50"))
        self.DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
        
        # Text Processing
        self.MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
        self.SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "1000"))
        self.SYNTHESIS_MAX_TOKENS = int(os.getenv("SYNTHESIS_MAX_TOKENS", "1500"))
        
        # Audio Configuration
        self.AUDIO_CLEANUP_DAYS = int(os.getenv("AUDIO_CLEANUP_DAYS", "7"))
        self.TTS_RATE = int(os.getenv("TTS_RATE", "180"))
        
        # Classification
        self.CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.3"))
        
        # Rate Limiting
        self.GROQ_RATE_LIMIT_DELAY = float(os.getenv("GROQ_RATE_LIMIT_DELAY", "0.5"))
        self.SEMANTIC_SCHOLAR_RATE_LIMIT = float(os.getenv("SEMANTIC_SCHOLAR_RATE_LIMIT", "1.0"))
        self.ARXIV_RATE_LIMIT = float(os.getenv("ARXIV_RATE_LIMIT", "0.5"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    def validate_configuration(self) -> dict:
        """
        Validate the current configuration and return status.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'api_keys_status': {}
        }
        
        # Check API keys
        if not self.GROQ_API_KEY:
            validation_results['errors'].append("GROQ_API_KEY is not configured")
            validation_results['valid'] = False
        
        validation_results['api_keys_status']['groq'] = bool(self.GROQ_API_KEY)
        
        # Check storage directories
        for dir_path in [self.STORAGE_BASE_DIR, self.PAPERS_DIR, self.AUDIO_DIR]:
            if not os.path.exists(dir_path):
                validation_results['warnings'].append(f"Directory does not exist: {dir_path}")
        
        # Check numeric values
        if self.MAX_CONCURRENT_REQUESTS <= 0:
            validation_results['errors'].append("MAX_CONCURRENT_REQUESTS must be positive")
            validation_results['valid'] = False
        
        if self.CLASSIFICATION_THRESHOLD < 0 or self.CLASSIFICATION_THRESHOLD > 1:
            validation_results['errors'].append("CLASSIFICATION_THRESHOLD must be between 0 and 1")
            validation_results['valid'] = False
        
        return validation_results
    
    def get_api_endpoints(self) -> dict:
        """
        Get API endpoint configuration.
        
        Returns:
            Dictionary with API endpoints
        """
        return {
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1',
            'arxiv': 'http://export.arxiv.org/api',
            'crossref': 'https://api.crossref.org/works',
            'groq': 'https://api.groq.com/openai/v1'
        }
    
    def get_file_paths(self) -> dict:
        """
        Get standardized file paths for the application.
        
        Returns:
            Dictionary with file paths
        """
        return {
            'storage_base': self.STORAGE_BASE_DIR,
            'papers': self.PAPERS_DIR,
            'audio': self.AUDIO_DIR,
            'logs': os.path.join(self.STORAGE_BASE_DIR, 'logs'),
            'temp': os.path.join(self.STORAGE_BASE_DIR, 'temp')
        }
    
    def update_from_dict(self, config_dict: dict):
        """
        Update configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'GROQ_API_KEY': '***' if self.GROQ_API_KEY else None,  # Hide API key
            'STORAGE_BASE_DIR': self.STORAGE_BASE_DIR,
            'PAPERS_DIR': self.PAPERS_DIR,
            'AUDIO_DIR': self.AUDIO_DIR,
            'MAX_CONCURRENT_REQUESTS': self.MAX_CONCURRENT_REQUESTS,
            'MAX_PAPERS_PER_SEARCH': self.MAX_PAPERS_PER_SEARCH,
            'DEFAULT_SEARCH_LIMIT': self.DEFAULT_SEARCH_LIMIT,
            'MAX_TEXT_LENGTH': self.MAX_TEXT_LENGTH,
            'SUMMARY_MAX_TOKENS': self.SUMMARY_MAX_TOKENS,
            'SYNTHESIS_MAX_TOKENS': self.SYNTHESIS_MAX_TOKENS,
            'AUDIO_CLEANUP_DAYS': self.AUDIO_CLEANUP_DAYS,
            'TTS_RATE': self.TTS_RATE,
            'CLASSIFICATION_THRESHOLD': self.CLASSIFICATION_THRESHOLD,
            'GROQ_RATE_LIMIT_DELAY': self.GROQ_RATE_LIMIT_DELAY,
            'SEMANTIC_SCHOLAR_RATE_LIMIT': self.SEMANTIC_SCHOLAR_RATE_LIMIT,
            'ARXIV_RATE_LIMIT': self.ARXIV_RATE_LIMIT,
            'LOG_LEVEL': self.LOG_LEVEL
        }
    
    @classmethod
    def create_default_config(cls) -> 'Config':
        """
        Create a default configuration instance.
        
        Returns:
            Default Config instance
        """
        return cls()
    
    def get_processing_limits(self) -> dict:
        """
        Get processing limits for different operations.
        
        Returns:
            Dictionary with processing limits
        """
        return {
            'max_concurrent_requests': self.MAX_CONCURRENT_REQUESTS,
            'max_papers_per_search': self.MAX_PAPERS_PER_SEARCH,
            'max_text_length': self.MAX_TEXT_LENGTH,
            'summary_max_tokens': self.SUMMARY_MAX_TOKENS,
            'synthesis_max_tokens': self.SYNTHESIS_MAX_TOKENS,
            'classification_threshold': self.CLASSIFICATION_THRESHOLD
        }
    
    def get_rate_limits(self) -> dict:
        """
        Get rate limiting configuration.
        
        Returns:
            Dictionary with rate limits
        """
        return {
            'groq_delay': self.GROQ_RATE_LIMIT_DELAY,
            'semantic_scholar_delay': self.SEMANTIC_SCHOLAR_RATE_LIMIT,
            'arxiv_delay': self.ARXIV_RATE_LIMIT
        }

# Global configuration instance
config = Config()
