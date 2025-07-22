"""
Topic Classification Agent - Categorizes papers using semantic similarity.
"""

import asyncio
from typing import Dict, List, Any, Optional
import numpy as np

# Try to import sentence_transformers, fall back if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    
# Import additional tools for fallback text similarity
import re
from collections import Counter
import math

from .base import BaseAgent, AgentResult
from utils.data_models import Paper

class TopicClassificationAgent(BaseAgent):
    """
    Agent responsible for classifying papers into user-defined topics using semantic similarity.
    
    Uses sentence transformers to create embeddings of paper content and topics,
    then finds the best matching topic based on cosine similarity.
    """
    
    def __init__(self):
        super().__init__("TopicClassification")
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.log_warning("sentence-transformers not available, using keyword matching only")
            self.model = None
            return
            
        try:
            self.log_info("Loading sentence transformer model...")
            # Use a lightweight, fast model suitable for semantic similarity
            if SentenceTransformer is not None:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.log_info("Sentence transformer model loaded successfully")
            else:
                self.model = None
        except Exception as e:
            self.log_error("Failed to load sentence transformer model", e)
            # Fall back to a simple keyword matching approach
            self.model = None
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Classify a paper into one of the provided topics.
        
        Args:
            input_data: Dictionary containing:
                - paper: Paper object to classify
                - topics: List of topic strings
                
        Returns:
            AgentResult containing the best matching topic
        """
        return await self.safe_execute("topic_classification", self._classify_paper, input_data)
    
    async def _classify_paper(self, input_data: Dict[str, Any]) -> str:
        """
        Main classification method.
        
        Args:
            input_data: Contains paper and topics
            
        Returns:
            Best matching topic string
        """
        paper = input_data['paper']
        topics = input_data['topics']
        
        if not topics:
            return "Unclassified"
        
        self.log_info(f"Classifying paper: {paper.title[:50]}...")
        
        # Use semantic similarity if model is available
        if self.model:
            topic = await self._classify_with_embeddings(paper, topics)
        else:
            # Fall back to keyword matching
            topic = self._classify_with_keywords(paper, topics)
        
        self.log_info(f"Paper classified as: {topic}")
        return topic
    
    async def _classify_with_embeddings(self, paper: Paper, topics: List[str]) -> str:
        """
        Classify paper using sentence embeddings and cosine similarity.
        
        Args:
            paper: Paper object to classify
            topics: List of topic strings
            
        Returns:
            Best matching topic
        """
        try:
            # Prepare text content for classification
            paper_content = self._prepare_paper_content(paper)
            
            # Generate embeddings
            # Run in thread pool to avoid blocking
            paper_embedding, topic_embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_embeddings, paper_content, topics
            )
            
            # Calculate similarities
            similarities = self._calculate_similarities(paper_embedding, topic_embeddings)
            
            # Find best matching topic
            best_topic_index = np.argmax(similarities)
            best_similarity = similarities[best_topic_index]
            
            # Use a threshold to determine if classification is confident enough
            if best_similarity > 0.3:  # Adjustable threshold
                return topics[best_topic_index]
            else:
                self.log_warning(f"Low confidence classification: {best_similarity:.3f}")
                return "Unclassified"
        
        except Exception as e:
            self.log_error("Embedding-based classification failed", e)
            # Fall back to keyword matching
            return self._classify_with_keywords(paper, topics)
    
    def _generate_embeddings(self, paper_content: str, topics: List[str]):
        """
        Generate embeddings for paper content and topics.
        
        Args:
            paper_content: Combined paper text
            topics: List of topics
            
        Returns:
            Tuple of (paper_embedding, topic_embeddings)
        """
        # Generate paper embedding
        if self.model is not None:
            paper_embedding = self.model.encode([paper_content])[0]
            # Generate topic embeddings
            topic_embeddings = self.model.encode(topics)
        else:
            raise ValueError("Model not initialized")
        
        return paper_embedding, topic_embeddings
    
    def _calculate_similarities(self, paper_embedding: np.ndarray, topic_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between paper and topics.
        
        Args:
            paper_embedding: Paper embedding vector
            topic_embeddings: Topic embedding vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize embeddings for cosine similarity
        paper_norm = paper_embedding / np.linalg.norm(paper_embedding)
        topic_norms = topic_embeddings / np.linalg.norm(topic_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarities
        similarities = np.dot(topic_norms, paper_norm)
        
        return similarities
    
    def _classify_with_keywords(self, paper: Paper, topics: List[str]) -> str:
        """
        Fallback classification method using keyword matching.
        
        Args:
            paper: Paper object to classify
            topics: List of topics
            
        Returns:
            Best matching topic based on keyword overlap
        """
        self.log_info("Using keyword-based classification fallback")
        
        paper_content = self._prepare_paper_content(paper).lower()
        paper_words = set(paper_content.split())
        
        best_topic = "Unclassified"
        best_score = 0
        
        for topic in topics:
            topic_words = set(topic.lower().split())
            
            # Calculate word overlap score
            overlap = len(paper_words.intersection(topic_words))
            
            # Normalize by topic length to avoid bias towards longer topics
            if len(topic_words) > 0:
                score = overlap / len(topic_words)
                
                if score > best_score:
                    best_score = score
                    best_topic = topic
        
        # Use a minimum threshold for keyword matching
        if best_score > 0.1:
            return best_topic
        else:
            return "Unclassified"
    
    def _prepare_paper_content(self, paper: Paper) -> str:
        """
        Prepare paper content for classification by combining relevant text fields.
        
        Args:
            paper: Paper object
            
        Returns:
            Combined text content
        """
        content_parts = []
        
        # Add title (weighted more heavily)
        if paper.title:
            content_parts.append(paper.title)
            content_parts.append(paper.title)  # Add twice for emphasis
        
        # Add abstract
        if paper.abstract:
            content_parts.append(paper.abstract)
        
        # Add venue if available (can indicate subject area)
        if paper.venue:
            content_parts.append(paper.venue)
        
        # Add full text if available (but limit to avoid overwhelming)
        if hasattr(paper, 'full_text') and paper.full_text:
            # Use first 1000 words of full text
            full_text_words = paper.full_text.split()[:1000]
            content_parts.append(' '.join(full_text_words))
        
        return ' '.join(content_parts)
    
    async def batch_classify(self, papers_and_topics: List[Dict[str, Any]]) -> List[str]:
        """
        Classify multiple papers efficiently.
        
        Args:
            papers_and_topics: List of dicts containing paper and topics
            
        Returns:
            List of classifications
        """
        self.log_info(f"Batch classifying {len(papers_and_topics)} papers")
        
        # Process papers concurrently
        tasks = []
        for item in papers_and_topics:
            task = self.process(item)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract successful classifications
        classifications = []
        for result in results:
            if isinstance(result, Exception):
                classifications.append("Unclassified")
            elif hasattr(result, 'success') and hasattr(result, 'data') and result.success:
                classifications.append(result.data)
            else:
                classifications.append("Unclassified")
        
        self.log_info("Batch classification completed")
        return classifications
    
    def get_topic_distribution(self, classifications: List[str]) -> Dict[str, int]:
        """
        Get distribution of papers across topics.
        
        Args:
            classifications: List of topic classifications
            
        Returns:
            Dictionary mapping topics to paper counts
        """
        distribution = {}
        
        for topic in classifications:
            distribution[topic] = distribution.get(topic, 0) + 1
        
        return distribution
    
    def get_classification_confidence_stats(self, paper: Paper, topics: List[str]) -> Dict[str, Any]:
        """
        Get detailed classification statistics for a paper.
        
        Args:
            paper: Paper to analyze
            topics: List of topics
            
        Returns:
            Dictionary with confidence scores for each topic or error message
        """
        if not self.model:
            return {"error": "Semantic model not available"}
        
        try:
            paper_content = self._prepare_paper_content(paper)
            
            # Generate embeddings
            paper_embedding = self.model.encode([paper_content])[0]
            topic_embeddings = self.model.encode(topics)
            
            # Calculate similarities
            similarities = self._calculate_similarities(paper_embedding, topic_embeddings)
            
            # Create confidence stats
            stats = {}
            for topic, similarity in zip(topics, similarities):
                stats[topic] = float(similarity)
            
            return stats
        
        except Exception as e:
            self.log_error("Failed to generate confidence stats", e)
            return {"error": str(e)}
