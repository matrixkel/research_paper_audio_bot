"""
Paper Discovery Agent - Searches for papers from various academic sources.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from urllib.parse import quote
import time

from .base import BaseAgent, AgentResult
from utils.data_models import Paper
from utils.config import Config

class PaperDiscoveryAgent(BaseAgent):
    """
    Agent responsible for discovering research papers from various sources.
    
    Supports:
    - Semantic Scholar API
    - ArXiv API
    - Filters for year, relevance, etc.
    """
    
    def __init__(self):
        super().__init__("PaperDiscovery")
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.arxiv_base = "http://export.arxiv.org/api"
        self.session = None
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Process paper discovery request.
        
        Args:
            input_data: Search parameters
            
        Returns:
            AgentResult containing discovered papers
        """
        return await self.safe_execute("paper_discovery", self._discover_papers, input_data)
    
    async def _discover_papers(self, search_params: Dict[str, Any]) -> List[Paper]:
        """
        Main paper discovery method that searches multiple sources.
        
        Args:
            search_params: Dictionary containing:
                - query: Search query string
                - max_papers: Maximum number of papers to retrieve
                - year_from: Starting year filter
                - year_to: Ending year filter
                - source: Source preference ("Both", "Semantic Scholar", "ArXiv")
                
        Returns:
            List of Paper objects
        """
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            query = search_params['query']
            max_papers = search_params.get('max_papers', 10)
            year_from = search_params.get('year_from', 2020)
            year_to = search_params.get('year_to', 2025)
            source = search_params.get('source', 'Both')
            
            papers = []
            
            # Search Semantic Scholar
            if source in ['Both', 'Semantic Scholar']:
                self.log_info(f"Searching Semantic Scholar for: {query}")
                semantic_papers = await self._search_semantic_scholar(query, max_papers//2 if source == 'Both' else max_papers, year_from, year_to)
                papers.extend(semantic_papers)
            
            # Search ArXiv
            if source in ['Both', 'ArXiv']:
                self.log_info(f"Searching ArXiv for: {query}")
                arxiv_papers = await self._search_arxiv(query, max_papers//2 if source == 'Both' else max_papers, year_from, year_to)
                papers.extend(arxiv_papers)
            
            # Remove duplicates based on title similarity
            papers = self._remove_duplicates(papers)
            
            # Limit to requested number
            papers = papers[:max_papers]
            
            self.log_info(f"Found {len(papers)} unique papers")
            return papers
    
    async def _search_semantic_scholar(self, query: str, max_papers: int, year_from: int, year_to: int) -> List[Paper]:
        """
        Search Semantic Scholar API for papers.
        
        Args:
            query: Search query
            max_papers: Maximum papers to retrieve
            year_from: Starting year
            year_to: Ending year
            
        Returns:
            List of Paper objects from Semantic Scholar
        """
        papers = []
        
        try:
            # Build the API request
            url = f"{self.semantic_scholar_base}/paper/search"
            params = {
                'query': query,
                'limit': min(max_papers, 100),  # API limit
                'fields': 'paperId,title,abstract,authors,year,publicationDate,citationCount,url,venue,externalIds'
            }
            
            # Add year filter if specified
            if year_from and year_to:
                params['year'] = f"{year_from}-{year_to}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data:
                        for item in data['data']:
                            try:
                                paper = self._parse_semantic_scholar_paper(item)
                                if paper:
                                    papers.append(paper)
                            except Exception as e:
                                self.log_warning(f"Failed to parse Semantic Scholar paper", error=str(e))
                                continue
                    
                    self.log_info(f"Retrieved {len(papers)} papers from Semantic Scholar")
                
                elif response.status == 429:  # Rate limited
                    self.log_warning("Semantic Scholar API rate limited, waiting...")
                    await asyncio.sleep(1)
                else:
                    self.log_error(f"Semantic Scholar API error: {response.status}")
        
        except Exception as e:
            self.log_error("Semantic Scholar search failed", e)
        
        return papers
    
    async def _search_arxiv(self, query: str, max_papers: int, year_from: int, year_to: int) -> List[Paper]:
        """
        Search ArXiv API for papers.
        
        Args:
            query: Search query
            max_papers: Maximum papers to retrieve
            year_from: Starting year
            year_to: Ending year
            
        Returns:
            List of Paper objects from ArXiv
        """
        papers = []
        
        try:
            # Build ArXiv query
            search_query = f"all:{quote(query)}"
            
            url = f"{self.arxiv_base}/query"
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': min(max_papers, 100),  # ArXiv limit
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    papers = self._parse_arxiv_response(xml_content, year_from, year_to)
                    self.log_info(f"Retrieved {len(papers)} papers from ArXiv")
                else:
                    self.log_error(f"ArXiv API error: {response.status}")
        
        except Exception as e:
            self.log_error("ArXiv search failed", e)
        
        return papers
    
    def _parse_semantic_scholar_paper(self, item: Dict) -> Optional[Paper]:
        """Parse a paper from Semantic Scholar API response"""
        try:
            # Extract authors
            authors = []
            if item.get('authors'):
                authors = [author.get('name', 'Unknown') for author in item['authors']]
            
            # Extract DOI
            doi = None
            external_ids = item.get('externalIds', {})
            if external_ids:
                doi = external_ids.get('DOI') or external_ids.get('ArXiv')
            
            # Create paper object
            paper = Paper(
                id=item.get('paperId', ''),
                title=item.get('title', 'Untitled'),
                abstract=item.get('abstract', ''),
                authors=authors,
                year=item.get('year'),
                source='Semantic Scholar',
                url=item.get('url'),
                doi=doi,
                venue=item.get('venue', {}).get('name') if item.get('venue') else None,
                citation_count=item.get('citationCount', 0)
            )
            
            return paper
            
        except Exception as e:
            self.log_error("Failed to parse Semantic Scholar paper", e, item_id=item.get('paperId'))
            return None
    
    def _parse_arxiv_response(self, xml_content: str, year_from: int, year_to: int) -> List[Paper]:
        """Parse ArXiv API XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom', 
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                try:
                    # Extract basic information
                    title = entry.find('atom:title', ns)
                    title_text = title.text.strip() if title is not None else 'Untitled'
                    
                    abstract = entry.find('atom:summary', ns)
                    abstract_text = abstract.text.strip() if abstract is not None else ''
                    
                    # Extract authors
                    authors = []
                    author_elements = entry.findall('atom:author', ns)
                    for author in author_elements:
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    # Extract publication date and year
                    published = entry.find('atom:published', ns)
                    year = None
                    if published is not None:
                        try:
                            year = int(published.text[:4])
                        except (ValueError, IndexError):
                            pass
                    
                    # Filter by year if specified
                    if year and (year < year_from or year > year_to):
                        continue
                    
                    # Extract ArXiv ID and URL
                    arxiv_id = None
                    url = None
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None:
                        url = id_elem.text
                        # Extract ArXiv ID from URL
                        if 'arxiv.org/abs/' in url:
                            arxiv_id = url.split('/')[-1]
                    
                    paper = Paper(
                        id=arxiv_id or url or title_text[:50],
                        title=title_text,
                        abstract=abstract_text,
                        authors=authors,
                        year=year,
                        source='ArXiv',
                        url=url,
                        doi=arxiv_id,
                        venue='ArXiv',
                        citation_count=0  # ArXiv doesn't provide citation counts
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    self.log_warning(f"Failed to parse ArXiv entry", error=str(e))
                    continue
        
        except ET.ParseError as e:
            self.log_error("Failed to parse ArXiv XML response", e)
        except Exception as e:
            self.log_error("Unexpected error parsing ArXiv response", e)
        
        return papers
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove duplicate papers based on title similarity.
        
        Args:
            papers: List of papers that may contain duplicates
            
        Returns:
            List of unique papers
        """
        if not papers:
            return papers
        
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison (lowercase, remove extra spaces)
            normalized_title = ' '.join(paper.title.lower().split())
            
            # Simple duplicate detection based on title similarity
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)
        
        removed_count = len(papers) - len(unique_papers)
        if removed_count > 0:
            self.log_info(f"Removed {removed_count} duplicate papers")
        
        return unique_papers
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """
        Check if two titles are similar enough to be considered duplicates.
        
        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if titles are similar enough to be duplicates
        """
        if title1 == title2:
            return True
        
        # Simple word-based similarity
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 and not words2:
            return True
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
