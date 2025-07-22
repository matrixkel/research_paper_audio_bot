"""
Paper Processing Agent - Extracts text and metadata from papers (PDFs, URLs, DOIs).
"""

import os
import asyncio
import aiohttp
import fitz  # PyMuPDF
import trafilatura
from typing import Dict, Any, Optional
import re
from urllib.parse import urlparse

from .base import BaseAgent, AgentResult
from utils.data_models import Paper
from utils.helpers import generate_paper_id

class PaperProcessingAgent(BaseAgent):
    """
    Agent responsible for processing papers from various sources and extracting text content.
    
    Supports:
    - PDF file processing (local files)
    - DOI resolution and content extraction
    - URL processing for academic papers
    - Text extraction and cleaning
    """
    
    def __init__(self):
        super().__init__("PaperProcessing")
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.crossref_base = "https://api.crossref.org/works"
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Process paper input based on type.
        
        Args:
            input_data: Dictionary containing:
                - type: 'pdf_file', 'doi', or 'url'
                - Additional fields based on type
                
        Returns:
            AgentResult containing Paper object
        """
        input_type = input_data.get('type')
        
        if input_type == 'pdf_file':
            return await self.safe_execute("pdf_processing", self._process_pdf_file, input_data['path'])
        elif input_type == 'doi':
            return await self.safe_execute("doi_processing", self._process_doi, input_data['doi'])
        elif input_type == 'url':
            return await self.safe_execute("url_processing", self._process_url, input_data['url'])
        else:
            return AgentResult(success=False, error=f"Unsupported input type: {input_type}")
    
    async def _process_pdf_file(self, file_path: str) -> Paper:
        """
        Process a local PDF file and extract text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Paper object with extracted content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        self.log_info(f"Processing PDF file: {file_path}")
        
        try:
            # Extract text using PyMuPDF
            text_content = self._extract_pdf_text(file_path)
            
            # Extract metadata from text
            metadata = self._extract_metadata_from_text(text_content)
            
            # Create paper object
            paper = Paper(
                id=generate_paper_id(),
                title=metadata.get('title', os.path.basename(file_path).replace('.pdf', '')),
                abstract=metadata.get('abstract', ''),
                authors=metadata.get('authors', []),
                year=metadata.get('year'),
                source='PDF Upload',
                url=None,
                doi=metadata.get('doi'),
                venue=metadata.get('venue'),
                citation_count=0,
                full_text=text_content
            )
            
            self.log_info(f"Successfully processed PDF: {paper.title}")
            return paper
            
        except Exception as e:
            self.log_error(f"Failed to process PDF file {file_path}", e)
            raise
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            
            # Clean up the text
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            self.log_error(f"Failed to extract text from PDF {file_path}", e)
            raise
    
    async def _process_doi(self, doi: str) -> Paper:
        """
        Process a DOI and fetch paper information.
        
        Args:
            doi: DOI identifier
            
        Returns:
            Paper object with fetched information
        """
        self.log_info(f"Processing DOI: {doi}")
        
        async with aiohttp.ClientSession() as session:
            # Try Semantic Scholar first
            paper = await self._fetch_from_semantic_scholar_doi(session, doi)
            
            if not paper:
                # Fall back to CrossRef
                paper = await self._fetch_from_crossref(session, doi)
            
            if not paper:
                # Create minimal paper object
                paper = Paper(
                    id=generate_paper_id(),
                    title=f"Paper with DOI: {doi}",
                    abstract="",
                    authors=[],
                    year=None,
                    source="DOI",
                    url=f"https://doi.org/{doi}",
                    doi=doi,
                    venue=None,
                    citation_count=0
                )
            
            self.log_info(f"Successfully processed DOI: {paper.title}")
            return paper
    
    async def _fetch_from_semantic_scholar_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Paper]:
        """Fetch paper information from Semantic Scholar using DOI"""
        try:
            url = f"{self.semantic_scholar_base}/paper/DOI:{doi}"
            params = {
                'fields': 'paperId,title,abstract,authors,year,publicationDate,citationCount,url,venue,externalIds'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_semantic_scholar_paper_response(data)
        
        except Exception as e:
            self.log_warning(f"Failed to fetch from Semantic Scholar for DOI {doi}", error=str(e))
        
        return None
    
    async def _fetch_from_crossref(self, session: aiohttp.ClientSession, doi: str) -> Optional[Paper]:
        """Fetch paper information from CrossRef"""
        try:
            url = f"{self.crossref_base}/{doi}"
            headers = {'User-Agent': 'ResearchPaperAnalysis/1.0 (mailto:researcher@example.com)'}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_crossref_response(data, doi)
        
        except Exception as e:
            self.log_warning(f"Failed to fetch from CrossRef for DOI {doi}", error=str(e))
        
        return None
    
    def _parse_semantic_scholar_paper_response(self, data: Dict) -> Optional[Paper]:
        """Parse Semantic Scholar API response"""
        try:
            authors = []
            if data.get('authors'):
                authors = [author.get('name', 'Unknown') for author in data['authors']]
            
            paper = Paper(
                id=data.get('paperId', generate_paper_id()),
                title=data.get('title', 'Untitled'),
                abstract=data.get('abstract', ''),
                authors=authors,
                year=data.get('year'),
                source='Semantic Scholar',
                url=data.get('url'),
                doi=data.get('externalIds', {}).get('DOI'),
                venue=data.get('venue', {}).get('name') if data.get('venue') else None,
                citation_count=data.get('citationCount', 0)
            )
            
            return paper
            
        except Exception as e:
            self.log_error("Failed to parse Semantic Scholar response", e)
            return None
    
    def _parse_crossref_response(self, data: Dict, doi: str) -> Optional[Paper]:
        """Parse CrossRef API response"""
        try:
            message = data.get('message', {})
            
            # Extract title
            title_list = message.get('title', [])
            title = title_list[0] if title_list else 'Untitled'
            
            # Extract authors
            authors = []
            author_list = message.get('author', [])
            for author in author_list:
                given = author.get('given', '')
                family = author.get('family', '')
                full_name = f"{given} {family}".strip()
                if full_name:
                    authors.append(full_name)
            
            # Extract year
            year = None
            date_parts = message.get('published-print', {}).get('date-parts') or \
                        message.get('published-online', {}).get('date-parts')
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
            
            # Extract venue/journal
            venue = message.get('container-title', [])
            venue_name = venue[0] if venue else None
            
            paper = Paper(
                id=generate_paper_id(),
                title=title,
                abstract=message.get('abstract', ''),
                authors=authors,
                year=year,
                source='CrossRef',
                url=f"https://doi.org/{doi}",
                doi=doi,
                venue=venue_name,
                citation_count=message.get('is-referenced-by-count', 0)
            )
            
            return paper
            
        except Exception as e:
            self.log_error("Failed to parse CrossRef response", e)
            return None
    
    async def _process_url(self, url: str) -> Paper:
        """
        Process a URL and extract paper content.
        
        Args:
            url: URL to the paper
            
        Returns:
            Paper object with extracted content
        """
        self.log_info(f"Processing URL: {url}")
        
        try:
            # Check if it's an ArXiv URL
            if 'arxiv.org' in url:
                paper = await self._process_arxiv_url(url)
            else:
                # Use web scraping for other URLs
                paper = await self._process_web_url(url)
            
            if not paper:
                # Create minimal paper object
                paper = Paper(
                    id=generate_paper_id(),
                    title=f"Paper from URL: {url}",
                    abstract="",
                    authors=[],
                    year=None,
                    source="URL",
                    url=url,
                    doi=None,
                    venue=None,
                    citation_count=0
                )
            
            self.log_info(f"Successfully processed URL: {paper.title}")
            return paper
            
        except Exception as e:
            self.log_error(f"Failed to process URL {url}", e)
            raise
    
    async def _process_arxiv_url(self, url: str) -> Optional[Paper]:
        """Process ArXiv URL to extract paper information"""
        try:
            # Extract ArXiv ID from URL
            arxiv_id = None
            if '/abs/' in url:
                arxiv_id = url.split('/abs/')[-1]
            elif '/pdf/' in url:
                arxiv_id = url.split('/pdf/')[-1].replace('.pdf', '')
            
            if not arxiv_id:
                return None
            
            # Fetch from ArXiv API
            async with aiohttp.ClientSession() as session:
                api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
                
                async with session.get(api_url) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        papers = self._parse_arxiv_xml(xml_content)
                        return papers[0] if papers else None
        
        except Exception as e:
            self.log_warning(f"Failed to process ArXiv URL {url}", error=str(e))
        
        return None
    
    def _parse_arxiv_xml(self, xml_content: str) -> list:
        """Parse ArXiv API XML response (reused from discovery agent)"""
        import xml.etree.ElementTree as ET
        
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                try:
                    title = entry.find('atom:title', ns)
                    title_text = title.text.strip() if title is not None else 'Untitled'
                    
                    abstract = entry.find('atom:summary', ns)
                    abstract_text = abstract.text.strip() if abstract is not None else ''
                    
                    authors = []
                    author_elements = entry.findall('atom:author', ns)
                    for author in author_elements:
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    published = entry.find('atom:published', ns)
                    year = None
                    if published is not None:
                        try:
                            year = int(published.text[:4])
                        except (ValueError, IndexError):
                            pass
                    
                    id_elem = entry.find('atom:id', ns)
                    url = id_elem.text if id_elem is not None else None
                    arxiv_id = url.split('/')[-1] if url else None
                    
                    paper = Paper(
                        id=arxiv_id or generate_paper_id(),
                        title=title_text,
                        abstract=abstract_text,
                        authors=authors,
                        year=year,
                        source='ArXiv',
                        url=url,
                        doi=arxiv_id,
                        venue='ArXiv',
                        citation_count=0
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    self.log_warning(f"Failed to parse ArXiv entry", error=str(e))
                    continue
        
        except ET.ParseError as e:
            self.log_error("Failed to parse ArXiv XML", e)
        
        return papers
    
    async def _process_web_url(self, url: str) -> Optional[Paper]:
        """Process general web URL using trafilatura"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        
                        # Extract text content using trafilatura
                        text_content = trafilatura.extract(html_content)
                        
                        if text_content:
                            # Extract metadata from text
                            metadata = self._extract_metadata_from_text(text_content)
                            
                            paper = Paper(
                                id=generate_paper_id(),
                                title=metadata.get('title', self._extract_title_from_url(url)),
                                abstract=metadata.get('abstract', text_content[:500] + '...'),
                                authors=metadata.get('authors', []),
                                year=metadata.get('year'),
                                source='Web',
                                url=url,
                                doi=metadata.get('doi'),
                                venue=metadata.get('venue'),
                                citation_count=0,
                                full_text=text_content
                            )
                            
                            return paper
        
        except Exception as e:
            self.log_warning(f"Failed to process web URL {url}", error=str(e))
        
        return None
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from URL"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if path:
            # Clean up the path to make it readable
            title = path.replace('-', ' ').replace('_', ' ').replace('/', ' - ')
            return title.title()
        
        return f"Document from {parsed.netloc}"
    
    def _extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text content using pattern matching.
        
        Args:
            text: Full text content
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {}
        
        # Clean text for processing
        text_lines = text.split('\n')
        first_500_words = ' '.join(text.split()[:500])
        
        # Extract title (usually first non-empty line or largest text block)
        title = self._extract_title_from_text(text_lines)
        if title:
            metadata['title'] = title
        
        # Extract abstract
        abstract = self._extract_abstract_from_text(text)
        if abstract:
            metadata['abstract'] = abstract
        
        # Extract authors
        authors = self._extract_authors_from_text(first_500_words)
        if authors:
            metadata['authors'] = authors
        
        # Extract year
        year = self._extract_year_from_text(first_500_words)
        if year:
            metadata['year'] = year
        
        # Extract DOI
        doi = self._extract_doi_from_text(text)
        if doi:
            metadata['doi'] = doi
        
        return metadata
    
    def _extract_title_from_text(self, lines: list) -> Optional[str]:
        """Extract title from text lines"""
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Skip common headers
                if not any(skip in line.lower() for skip in ['abstract', 'introduction', 'keywords', 'doi:', 'arxiv:']):
                    return line
        return None
    
    def _extract_abstract_from_text(self, text: str) -> Optional[str]:
        """Extract abstract from text"""
        # Look for abstract section
        abstract_patterns = [
            r'abstract[:\s]+(.{50,1000}?)(?:\n\n|\n[A-Z]|\nKeywords|\nIntroduction)',
            r'Abstract[:\s]+(.{50,1000}?)(?:\n\n|\n[A-Z]|\nKeywords|\nIntroduction)',
            r'ABSTRACT[:\s]+(.{50,1000}?)(?:\n\n|\n[A-Z]|\nKEYWORDS|\nINTRODUCTION)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                if len(abstract) > 50:
                    return abstract
        
        return None
    
    def _extract_authors_from_text(self, text: str) -> list:
        """Extract authors from text"""
        authors = []
        
        # Common author patterns
        patterns = [
            r'Authors?[:\s]+(.{10,200}?)(?:\n|Abstract|Keywords)',
            r'By[:\s]+(.{10,200}?)(?:\n|Abstract|Keywords)',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Parse individual author names
                if isinstance(match, str):
                    author_names = [name.strip() for name in re.split(r'[,;]|and', match)]
                    for name in author_names:
                        name = name.strip()
                        if len(name) > 3 and len(name) < 50:
                            authors.append(name)
                
                if len(authors) >= 10:  # Reasonable limit
                    break
        
        return list(set(authors))  # Remove duplicates
    
    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract publication year from text"""
        # Look for 4-digit years in reasonable range
        year_pattern = r'\b(19[8-9]\d|20[0-2]\d)\b'
        matches = re.findall(year_pattern, text)
        
        if matches:
            # Return the most recent year found
            years = [int(year) for year in matches]
            return max(years)
        
        return None
    
    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract DOI from text"""
        doi_pattern = r'(?:DOI|doi)[:\s]*(10\.\d+/[^\s\]]+)'
        match = re.search(doi_pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'\f', ' ', text)  # Form feed
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control characters
        
        return text.strip()
