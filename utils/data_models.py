"""
Data models for the Multi-Agent Research Paper Analysis System.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

@dataclass
class Paper:
    """
    Data model representing a research paper with all its metadata.
    """
    id: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    source: str = ""  # Where the paper was found (ArXiv, Semantic Scholar, etc.)
    url: Optional[str] = None
    doi: Optional[str] = None
    venue: Optional[str] = None  # Journal, Conference, etc.
    citation_count: int = 0
    full_text: Optional[str] = None  # Full paper text if available
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization validation and cleanup"""
        # Ensure ID is not empty
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Clean up title
        if self.title:
            self.title = self.title.strip()
        
        # Clean up authors list
        self.authors = [author.strip() for author in self.authors if author.strip()]
        
        # Validate year
        if self.year is not None:
            current_year = datetime.now().year
            if self.year < 1900 or self.year > current_year + 1:
                self.year = None
    
    def get_display_title(self, max_length: int = 100) -> str:
        """Get a display-friendly title with length limit"""
        if not self.title:
            return "Untitled Paper"
        
        if len(self.title) <= max_length:
            return self.title
        
        return self.title[:max_length - 3] + "..."
    
    def get_author_string(self, max_authors: int = 3) -> str:
        """Get a formatted author string"""
        if not self.authors:
            return "Unknown Authors"
        
        if len(self.authors) <= max_authors:
            if len(self.authors) == 1:
                return self.authors[0]
            elif len(self.authors) == 2:
                return f"{self.authors[0]} and {self.authors[1]}"
            else:
                return ", ".join(self.authors[:-1]) + f", and {self.authors[-1]}"
        else:
            return ", ".join(self.authors[:max_authors]) + " et al."
    
    def has_full_text(self) -> bool:
        """Check if full text is available"""
        return bool(self.full_text and self.full_text.strip())
    
    def get_content_for_analysis(self) -> str:
        """Get combined content for analysis (title + abstract + full_text)"""
        content_parts = []
        
        if self.title:
            content_parts.append(self.title)
        
        if self.abstract:
            content_parts.append(self.abstract)
        
        if self.has_full_text():
            # Limit full text to reasonable length
            full_text = self.full_text[:5000]  # First 5000 characters
            content_parts.append(full_text)
        
        return " ".join(content_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'year': self.year,
            'source': self.source,
            'url': self.url,
            'doi': self.doi,
            'venue': self.venue,
            'citation_count': self.citation_count,
            'full_text': self.full_text,
            'keywords': self.keywords,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

@dataclass
class ProcessingResult:
    """
    Data model representing the result of processing a paper through the analysis pipeline.
    """
    paper_id: str
    topic: str
    summary: str
    synthesis: Optional[str] = None
    confidence_score: float = 0.0
    processing_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Ensure confidence score is within valid range
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        
        # Clean up topic
        if self.topic:
            self.topic = self.topic.strip()
        else:
            self.topic = "Unclassified"
    
    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return bool(self.summary and not self.errors)
    
    def has_synthesis(self) -> bool:
        """Check if synthesis is available"""
        return bool(self.synthesis and self.synthesis.strip())
    
    def get_summary_preview(self, max_length: int = 200) -> str:
        """Get a preview of the summary"""
        if not self.summary:
            return "No summary available"
        
        if len(self.summary) <= max_length:
            return self.summary
        
        return self.summary[:max_length - 3] + "..."
    
    def add_error(self, error: str):
        """Add an error to the errors list"""
        if error and error not in self.errors:
            self.errors.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'paper_id': self.paper_id,
            'topic': self.topic,
            'summary': self.summary,
            'synthesis': self.synthesis,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'errors': self.errors,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

@dataclass
class AudioResult:
    """
    Data model representing the result of text-to-speech conversion.
    """
    paper_id: str
    file_path: str
    filename: str
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    format: str = "mp3"
    engine_used: str = ""
    quality: str = "standard"  # standard, high, low
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Ensure duration is non-negative
        self.duration_seconds = max(0.0, self.duration_seconds)
        
        # Ensure file size is non-negative
        self.file_size_bytes = max(0, self.file_size_bytes)
    
    def get_duration_string(self) -> str:
        """Get human-readable duration string"""
        if self.duration_seconds < 60:
            return f"{int(self.duration_seconds)}s"
        elif self.duration_seconds < 3600:
            minutes = int(self.duration_seconds // 60)
            seconds = int(self.duration_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(self.duration_seconds // 3600)
            minutes = int((self.duration_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_file_size_string(self) -> str:
        """Get human-readable file size string"""
        if self.file_size_bytes < 1024:
            return f"{self.file_size_bytes} B"
        elif self.file_size_bytes < 1024 * 1024:
            kb = self.file_size_bytes / 1024
            return f"{kb:.1f} KB"
        else:
            mb = self.file_size_bytes / (1024 * 1024)
            return f"{mb:.1f} MB"
    
    def is_valid(self) -> bool:
        """Check if the audio result is valid"""
        import os
        return os.path.exists(self.file_path) and self.file_size_bytes > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'paper_id': self.paper_id,
            'file_path': self.file_path,
            'filename': self.filename,
            'duration_seconds': self.duration_seconds,
            'file_size_bytes': self.file_size_bytes,
            'format': self.format,
            'engine_used': self.engine_used,
            'quality': self.quality,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata
        }

@dataclass
class SearchQuery:
    """
    Data model representing a search query with filters and parameters.
    """
    query: str
    max_papers: int = 10
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    sources: List[str] = field(default_factory=lambda: ["Semantic Scholar", "ArXiv"])
    sort_by: str = "relevance"  # relevance, date, citations
    include_preprints: bool = True
    min_citation_count: int = 0
    topics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Clean up query
        self.query = self.query.strip() if self.query else ""
        
        # Validate max_papers
        self.max_papers = max(1, min(100, self.max_papers))
        
        # Validate years
        current_year = datetime.now().year
        if self.year_from is not None:
            self.year_from = max(1900, min(current_year, self.year_from))
        if self.year_to is not None:
            self.year_to = max(1900, min(current_year, self.year_to))
        
        # Ensure year_from <= year_to
        if self.year_from and self.year_to and self.year_from > self.year_to:
            self.year_from, self.year_to = self.year_to, self.year_from
        
        # Validate min_citation_count
        self.min_citation_count = max(0, self.min_citation_count)
    
    def is_valid(self) -> bool:
        """Check if the search query is valid"""
        return bool(self.query and self.query.strip())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'query': self.query,
            'max_papers': self.max_papers,
            'year_from': self.year_from,
            'year_to': self.year_to,
            'sources': self.sources,
            'sort_by': self.sort_by,
            'include_preprints': self.include_preprints,
            'min_citation_count': self.min_citation_count,
            'topics': self.topics,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

@dataclass
class SystemStats:
    """
    Data model for system statistics and metrics.
    """
    total_papers_processed: int = 0
    total_summaries_generated: int = 0
    total_syntheses_generated: int = 0
    total_audio_files_created: int = 0
    total_citations_generated: int = 0
    processing_time_avg: float = 0.0
    storage_used_mb: float = 0.0
    active_topics: List[str] = field(default_factory=list)
    most_cited_papers: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_stats(self, **kwargs):
        """Update statistics with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_papers_processed': self.total_papers_processed,
            'total_summaries_generated': self.total_summaries_generated,
            'total_syntheses_generated': self.total_syntheses_generated,
            'total_audio_files_created': self.total_audio_files_created,
            'total_citations_generated': self.total_citations_generated,
            'processing_time_avg': self.processing_time_avg,
            'storage_used_mb': self.storage_used_mb,
            'active_topics': self.active_topics,
            'most_cited_papers': self.most_cited_papers,
            'success_rate': self.success_rate,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

# Utility functions for data model operations

def create_paper_from_dict(data: Dict[str, Any]) -> Paper:
    """Create a Paper object from dictionary data"""
    # Handle datetime conversion
    created_at = data.get('created_at')
    if created_at and isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return Paper(
        id=data.get('id', str(uuid.uuid4())),
        title=data.get('title', ''),
        abstract=data.get('abstract', ''),
        authors=data.get('authors', []),
        year=data.get('year'),
        source=data.get('source', ''),
        url=data.get('url'),
        doi=data.get('doi'),
        venue=data.get('venue'),
        citation_count=data.get('citation_count', 0),
        full_text=data.get('full_text'),
        keywords=data.get('keywords', []),
        created_at=created_at or datetime.now()
    )

def create_processing_result_from_dict(data: Dict[str, Any]) -> ProcessingResult:
    """Create a ProcessingResult object from dictionary data"""
    # Handle datetime conversion
    created_at = data.get('created_at')
    if created_at and isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return ProcessingResult(
        paper_id=data.get('paper_id', ''),
        topic=data.get('topic', ''),
        summary=data.get('summary', ''),
        synthesis=data.get('synthesis'),
        confidence_score=data.get('confidence_score', 0.0),
        processing_time=data.get('processing_time'),
        errors=data.get('errors', []),
        metadata=data.get('metadata', {}),
        created_at=created_at or datetime.now()
    )

def create_audio_result_from_dict(data: Dict[str, Any]) -> AudioResult:
    """Create an AudioResult object from dictionary data"""
    # Handle datetime conversion
    created_at = data.get('created_at')
    if created_at and isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return AudioResult(
        paper_id=data.get('paper_id', ''),
        file_path=data.get('file_path', ''),
        filename=data.get('filename', ''),
        duration_seconds=data.get('duration_seconds', 0.0),
        file_size_bytes=data.get('file_size_bytes', 0),
        format=data.get('format', 'mp3'),
        engine_used=data.get('engine_used', ''),
        quality=data.get('quality', 'standard'),
        created_at=created_at or datetime.now(),
        metadata=data.get('metadata', {})
    )
