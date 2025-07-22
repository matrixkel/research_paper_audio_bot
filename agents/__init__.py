"""
Multi-Agent Research Paper Analysis System

This package contains specialized agents for different aspects of research paper processing:
- CoordinatorAgent: Orchestrates the entire workflow
- PaperDiscoveryAgent: Searches and discovers papers from various sources
- PaperProcessingAgent: Extracts text and metadata from papers
- TopicClassificationAgent: Categorizes papers by research topics
- SummarizationAgent: Generates individual paper summaries
- SynthesisAgent: Creates cross-paper topic syntheses
- TextToSpeechAgent: Converts text to audio
- CitationManagerAgent: Manages citations and references
"""

from .base import BaseAgent
from .coordinator import CoordinatorAgent
from .paper_discovery import PaperDiscoveryAgent
from .paper_processing import PaperProcessingAgent
from .topic_classification import TopicClassificationAgent
from .summarization import SummarizationAgent
from .synthesis import SynthesisAgent
from .text_to_speech import TextToSpeechAgent
from .citation_manager import CitationManagerAgent

__all__ = [
    'BaseAgent',
    'CoordinatorAgent',
    'PaperDiscoveryAgent',
    'PaperProcessingAgent',
    'TopicClassificationAgent',
    'SummarizationAgent',
    'SynthesisAgent',
    'TextToSpeechAgent',
    'CitationManagerAgent'
]
