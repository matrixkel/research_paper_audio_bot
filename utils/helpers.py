"""
Helper utilities for the Multi-Agent Research Paper Analysis System.
"""

import os
import uuid
import hashlib
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

def generate_paper_id(title: str = None, authors: List[str] = None) -> str:
    """
    Generate a unique ID for a paper based on its metadata.
    
    Args:
        title: Paper title
        authors: List of authors
        
    Returns:
        Unique paper ID
    """
    if title or authors:
        # Create a deterministic ID based on content
        content = ""
        if title:
            content += title.lower().strip()
        if authors:
            content += "".join(sorted([author.lower().strip() for author in authors]))
        
        # Create hash of content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"paper_{content_hash}"
    else:
        # Generate random ID
        return f"paper_{uuid.uuid4().hex[:12]}"

def ensure_directories():
    """
    Ensure all required directories exist for the application.
    """
    directories = [
        "storage",
        "storage/papers",
        "storage/audio",
        "storage/logs",
        "storage/temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def clean_filename(filename: str) -> str:
    """
    Clean a filename to be safe for filesystem use.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive whitespace
    filename = re.sub(r'\s+', '_', filename)
    
    # Limit length
    if len(filename) > 200:
        name_part = filename[:190]
        extension = filename[-10:] if '.' in filename[-10:] else ''
        filename = name_part + extension
    
    return filename.strip('._')

def sanitize_text(text: str) -> str:
    """
    Sanitize text content for processing and display.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
        'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'about', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
        'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'research', 'study', 'paper', 'analysis', 'method', 'approach',
        'results', 'findings', 'data', 'using', 'used', 'based', 'show',
        'shows', 'found', 'also', 'however', 'therefore', 'thus', 'such'
    }
    
    # Filter out stop words and count frequency
    from collections import Counter
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_counts = Counter(filtered_words)
    
    # Return most common words
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    return keywords

def truncate_text(text: str, max_length: int, preserve_words: bool = True) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        preserve_words: Whether to preserve word boundaries
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    if preserve_words:
        # Find the last space before the limit
        truncate_point = text.rfind(' ', 0, max_length - 3)
        if truncate_point > max_length // 2:  # Ensure we don't truncate too much
            return text[:truncate_point] + "..."
    
    return text[:max_length - 3] + "..."

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to a human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        kb = bytes_size / 1024
        return f"{kb:.1f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        mb = bytes_size / (1024 * 1024)
        return f"{mb:.1f} MB"
    else:
        gb = bytes_size / (1024 * 1024 * 1024)
        return f"{gb:.1f} GB"

def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))

def validate_doi(doi: str) -> bool:
    """
    Validate if a string is a valid DOI.
    
    Args:
        doi: DOI string to validate
        
    Returns:
        True if valid DOI, False otherwise
    """
    doi_pattern = re.compile(r'^10\.\d{4,}/[^\s]+$')
    return bool(doi_pattern.match(doi.strip()))

def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract DOI from text content.
    
    Args:
        text: Text to search for DOI
        
    Returns:
        Extracted DOI or None
    """
    doi_patterns = [
        r'(?:DOI|doi)[:\s]*(10\.\d{4,}/[^\s\]]+)',
        r'https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s\]]+)',
        r'\b(10\.\d{4,}/[^\s\]]+)\b'
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1)
            if validate_doi(doi):
                return doi
    
    return None

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: File name
        
    Returns:
        File extension (without dot)
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')

def is_pdf_file(filename: str) -> bool:
    """
    Check if filename represents a PDF file.
    
    Args:
        filename: File name to check
        
    Returns:
        True if PDF file, False otherwise
    """
    return get_file_extension(filename) == 'pdf'

def create_safe_filename(title: str, max_length: int = 100) -> str:
    """
    Create a safe filename from a title.
    
    Args:
        title: Paper title or similar text
        max_length: Maximum filename length
        
    Returns:
        Safe filename
    """
    if not title:
        return f"document_{uuid.uuid4().hex[:8]}"
    
    # Clean the title
    filename = re.sub(r'[^\w\s-]', '', title)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')
    
    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # Ensure it's not empty
    if not filename:
        filename = f"document_{uuid.uuid4().hex[:8]}"
    
    return filename

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("ResearchPaperAnalysis")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for diagnostics.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_usage_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
            'timestamp': datetime.now().isoformat()
        }
    except ImportError:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat(),
            'note': 'psutil not available for detailed system info'
        }

async def run_with_timeout(coro, timeout_seconds: float):
    """
    Run a coroutine with a timeout.
    
    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        
    Returns:
        Result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def get_storage_usage(directory: str) -> Dict[str, Any]:
    """
    Get storage usage statistics for a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Dictionary with storage usage information
    """
    try:
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_count': file_count,
            'directory': directory
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'directory': directory
        }

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to end at a word boundary
        if end < len(text):
            # Find the last space before the end
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunks.append(text[start:end])
        
        # Move start forward, accounting for overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later ones taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result
