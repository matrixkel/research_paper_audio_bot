"""
Citation Manager Agent - Manages citations and references in various formats.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from .base import BaseAgent, AgentResult
from utils.data_models import Paper

class CitationManagerAgent(BaseAgent):
    """
    Agent responsible for managing citations and generating formatted references.
    
    Supports multiple citation formats:
    - APA (American Psychological Association)
    - MLA (Modern Language Association)
    - Chicago (Chicago Manual of Style)
    - BibTeX (LaTeX bibliography format)
    
    Handles citation generation, validation, and formatting.
    """
    
    def __init__(self):
        super().__init__("CitationManager")
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Generate citations for papers in specified format.
        
        Args:
            input_data: Dictionary containing:
                - papers: List of Paper objects
                - format: Citation format ('APA', 'MLA', 'Chicago', 'BibTeX')
                
        Returns:
            AgentResult containing list of formatted citations
        """
        return await self.safe_execute("citation_generation", self._generate_citations, input_data)
    
    async def _generate_citations(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Generate citations for all papers in the specified format.
        
        Args:
            input_data: Contains papers and format
            
        Returns:
            List of formatted citation strings
        """
        papers = input_data['papers']
        citation_format = input_data['format'].upper()
        
        self.log_info(f"Generating {citation_format} citations for {len(papers)} papers")
        
        citations = []
        
        for paper in papers:
            try:
                if citation_format == 'APA':
                    citation = self._format_apa_citation(paper)
                elif citation_format == 'MLA':
                    citation = self._format_mla_citation(paper)
                elif citation_format == 'CHICAGO':
                    citation = self._format_chicago_citation(paper)
                elif citation_format == 'BIBTEX':
                    citation = self._format_bibtex_citation(paper)
                else:
                    citation = f"Unsupported format: {citation_format}"
                
                citations.append(citation)
                
            except Exception as e:
                self.log_error(f"Failed to format citation for paper {paper.id}", e)
                citations.append(f"Error formatting citation for: {paper.title}")
        
        self.log_info(f"Successfully generated {len(citations)} citations")
        return citations
    
    def _format_apa_citation(self, paper: Paper) -> str:
        """
        Format citation in APA style.
        
        Args:
            paper: Paper object
            
        Returns:
            APA formatted citation
        """
        citation_parts = []
        
        # Authors
        if paper.authors:
            author_str = self._format_apa_authors(paper.authors)
            citation_parts.append(author_str)
        else:
            citation_parts.append("[No author]")
        
        # Year
        year_str = f"({paper.year})" if paper.year else "(n.d.)"
        citation_parts.append(year_str)
        
        # Title
        title = self._clean_title(paper.title)
        citation_parts.append(f"{title}.")
        
        # Venue/Journal
        if paper.venue:
            venue_str = f"*{paper.venue}*"
            citation_parts.append(venue_str)
        
        # DOI or URL
        if paper.doi:
            citation_parts.append(f"https://doi.org/{paper.doi}")
        elif paper.url:
            citation_parts.append(paper.url)
        
        return " ".join(citation_parts)
    
    def _format_mla_citation(self, paper: Paper) -> str:
        """
        Format citation in MLA style.
        
        Args:
            paper: Paper object
            
        Returns:
            MLA formatted citation
        """
        citation_parts = []
        
        # Authors
        if paper.authors:
            author_str = self._format_mla_authors(paper.authors)
            citation_parts.append(f"{author_str}.")
        
        # Title
        title = self._clean_title(paper.title)
        citation_parts.append(f'"{title}."')
        
        # Venue
        if paper.venue:
            citation_parts.append(f"*{paper.venue}*,")
        
        # Year
        if paper.year:
            citation_parts.append(f"{paper.year},")
        
        # DOI or URL
        if paper.doi:
            citation_parts.append(f"doi:{paper.doi}.")
        elif paper.url:
            citation_parts.append(f"{paper.url}.")
        
        return " ".join(citation_parts)
    
    def _format_chicago_citation(self, paper: Paper) -> str:
        """
        Format citation in Chicago style.
        
        Args:
            paper: Paper object
            
        Returns:
            Chicago formatted citation
        """
        citation_parts = []
        
        # Authors
        if paper.authors:
            author_str = self._format_chicago_authors(paper.authors)
            citation_parts.append(f"{author_str}.")
        
        # Title
        title = self._clean_title(paper.title)
        citation_parts.append(f'"{title}."')
        
        # Venue
        if paper.venue:
            citation_parts.append(f"*{paper.venue}*")
        
        # Year
        if paper.year:
            citation_parts.append(f"({paper.year}).")
        
        # DOI or URL
        if paper.doi:
            citation_parts.append(f"https://doi.org/{paper.doi}.")
        elif paper.url:
            citation_parts.append(f"Accessed {datetime.now().strftime('%B %d, %Y')}. {paper.url}.")
        
        return " ".join(citation_parts)
    
    def _format_bibtex_citation(self, paper: Paper) -> str:
        """
        Format citation in BibTeX style.
        
        Args:
            paper: Paper object
            
        Returns:
            BibTeX formatted citation
        """
        # Generate citation key
        key = self._generate_bibtex_key(paper)
        
        # Determine entry type
        entry_type = "article" if paper.venue else "misc"
        
        citation_lines = [f"@{entry_type}{{{key},"]
        
        # Title
        title = self._clean_title(paper.title)
        citation_lines.append(f"  title={{{title}}},")
        
        # Authors
        if paper.authors:
            authors_str = " and ".join(paper.authors)
            citation_lines.append(f"  author={{{authors_str}}},")
        
        # Venue
        if paper.venue:
            if entry_type == "article":
                citation_lines.append(f"  journal={{{paper.venue}}},")
            else:
                citation_lines.append(f"  howpublished={{{paper.venue}}},")
        
        # Year
        if paper.year:
            citation_lines.append(f"  year={{{paper.year}}},")
        
        # DOI
        if paper.doi:
            citation_lines.append(f"  doi={{{paper.doi}}},")
        
        # URL
        if paper.url:
            citation_lines.append(f"  url={{{paper.url}}},")
        
        citation_lines.append("}")
        
        return "\n".join(citation_lines)
    
    def _format_apa_authors(self, authors: List[str]) -> str:
        """Format authors for APA style"""
        if not authors:
            return ""
        
        formatted_authors = []
        for author in authors[:7]:  # APA limits to 7 authors
            # Try to format as "Last, F. M."
            formatted_author = self._format_author_name(author, style="apa")
            formatted_authors.append(formatted_author)
        
        if len(authors) == 1:
            return formatted_authors[0]
        elif len(authors) == 2:
            return f"{formatted_authors[0]} & {formatted_authors[1]}"
        elif len(authors) <= 7:
            return ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
        else:
            return ", ".join(formatted_authors[:6]) + ", ... " + formatted_authors[-1]
    
    def _format_mla_authors(self, authors: List[str]) -> str:
        """Format authors for MLA style"""
        if not authors:
            return ""
        
        if len(authors) == 1:
            return self._format_author_name(authors[0], style="mla")
        elif len(authors) == 2:
            first = self._format_author_name(authors[0], style="mla")
            second = self._format_author_name(authors[1], style="mla_additional")
            return f"{first} and {second}"
        else:
            first = self._format_author_name(authors[0], style="mla")
            return f"{first} et al"
    
    def _format_chicago_authors(self, authors: List[str]) -> str:
        """Format authors for Chicago style"""
        if not authors:
            return ""
        
        formatted_authors = []
        for i, author in enumerate(authors[:10]):  # Reasonable limit
            if i == 0:
                # First author: Last, First
                formatted_author = self._format_author_name(author, style="chicago_first")
            else:
                # Additional authors: First Last
                formatted_author = self._format_author_name(author, style="chicago_additional")
            formatted_authors.append(formatted_author)
        
        if len(authors) == 1:
            return formatted_authors[0]
        elif len(authors) == 2:
            return f"{formatted_authors[0]} and {formatted_authors[1]}"
        elif len(authors) <= 10:
            return ", ".join(formatted_authors[:-1]) + f", and {formatted_authors[-1]}"
        else:
            return ", ".join(formatted_authors[:10]) + " et al"
    
    def _format_author_name(self, name: str, style: str) -> str:
        """
        Format individual author name based on citation style.
        
        Args:
            name: Author name
            style: Formatting style
            
        Returns:
            Formatted author name
        """
        name = name.strip()
        
        # Handle "First Last" format
        parts = name.split()
        
        if len(parts) < 2:
            return name  # Return as-is if can't parse
        
        if style == "apa":
            # "Last, F. M."
            last = parts[-1]
            first_initials = [p[0] + "." for p in parts[:-1] if p]
            return f"{last}, {' '.join(first_initials)}"
        
        elif style == "mla":
            # "Last, First"
            last = parts[-1]
            first_parts = parts[:-1]
            return f"{last}, {' '.join(first_parts)}"
        
        elif style == "mla_additional":
            # "First Last"
            return name
        
        elif style == "chicago_first":
            # "Last, First"
            last = parts[-1]
            first_parts = parts[:-1]
            return f"{last}, {' '.join(first_parts)}"
        
        elif style == "chicago_additional":
            # "First Last"
            return name
        
        else:
            return name
    
    def _generate_bibtex_key(self, paper: Paper) -> str:
        """
        Generate a BibTeX citation key.
        
        Args:
            paper: Paper object
            
        Returns:
            BibTeX key string
        """
        # Start with first author's last name
        key_parts = []
        
        if paper.authors:
            first_author = paper.authors[0]
            author_parts = first_author.split()
            if author_parts:
                last_name = author_parts[-1].lower()
                # Remove non-alphanumeric characters
                last_name = re.sub(r'[^a-zA-Z0-9]', '', last_name)
                key_parts.append(last_name)
        
        # Add year
        if paper.year:
            key_parts.append(str(paper.year))
        
        # Add first significant word from title
        if paper.title:
            title_words = paper.title.split()
            for word in title_words:
                clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
                if len(clean_word) > 3 and clean_word not in ['the', 'and', 'for', 'with']:
                    key_parts.append(clean_word[:8])  # Limit length
                    break
        
        # Join with underscore or use fallback
        if key_parts:
            return "_".join(key_parts)
        else:
            return f"paper_{paper.id}" if paper.id else "unknown_paper"
    
    def _clean_title(self, title: str) -> str:
        """
        Clean title for citation formatting.
        
        Args:
            title: Original title
            
        Returns:
            Cleaned title
        """
        if not title:
            return "Untitled"
        
        # Remove extra whitespace
        title = " ".join(title.split())
        
        # Remove trailing periods
        title = title.rstrip('.')
        
        return title
    
    async def validate_citations(self, citations: List[str], format_type: str) -> Dict[str, Any]:
        """
        Validate generated citations for completeness and format compliance.
        
        Args:
            citations: List of citation strings
            format_type: Citation format
            
        Returns:
            Validation report
        """
        self.log_info(f"Validating {len(citations)} {format_type} citations")
        
        validation_report = {
            'total_citations': len(citations),
            'valid_citations': 0,
            'issues': [],
            'format': format_type
        }
        
        for i, citation in enumerate(citations):
            issues = self._validate_single_citation(citation, format_type)
            
            if not issues:
                validation_report['valid_citations'] += 1
            else:
                validation_report['issues'].append({
                    'citation_index': i,
                    'issues': issues,
                    'citation_preview': citation[:100] + '...' if len(citation) > 100 else citation
                })
        
        validation_report['validation_success_rate'] = (
            validation_report['valid_citations'] / validation_report['total_citations']
            if validation_report['total_citations'] > 0 else 0
        )
        
        return validation_report
    
    def _validate_single_citation(self, citation: str, format_type: str) -> List[str]:
        """
        Validate a single citation for format compliance.
        
        Args:
            citation: Citation string
            format_type: Citation format
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not citation or not citation.strip():
            issues.append("Empty citation")
            return issues
        
        citation = citation.strip()
        
        # Common validation checks
        if len(citation) < 20:
            issues.append("Citation appears too short")
        
        if format_type.upper() == 'APA':
            issues.extend(self._validate_apa_citation(citation))
        elif format_type.upper() == 'MLA':
            issues.extend(self._validate_mla_citation(citation))
        elif format_type.upper() == 'CHICAGO':
            issues.extend(self._validate_chicago_citation(citation))
        elif format_type.upper() == 'BIBTEX':
            issues.extend(self._validate_bibtex_citation(citation))
        
        return issues
    
    def _validate_apa_citation(self, citation: str) -> List[str]:
        """Validate APA format specific requirements"""
        issues = []
        
        # Check for year in parentheses
        if not re.search(r'\(\d{4}\)', citation) and '(n.d.)' not in citation:
            issues.append("Missing or malformed publication year")
        
        # Check for DOI or URL
        if 'doi.org' not in citation and 'http' not in citation:
            issues.append("Missing DOI or URL (recommended for APA)")
        
        return issues
    
    def _validate_mla_citation(self, citation: str) -> List[str]:
        """Validate MLA format specific requirements"""
        issues = []
        
        # Check for quoted title
        if '"' not in citation:
            issues.append("Title should be in quotation marks for MLA")
        
        return issues
    
    def _validate_chicago_citation(self, citation: str) -> List[str]:
        """Validate Chicago format specific requirements"""
        issues = []
        
        # Basic Chicago format check
        if not re.search(r'\(\d{4}\)', citation):
            issues.append("Missing publication year in parentheses")
        
        return issues
    
    def _validate_bibtex_citation(self, citation: str) -> List[str]:
        """Validate BibTeX format specific requirements"""
        issues = []
        
        # Check BibTeX structure
        if not citation.startswith('@'):
            issues.append("BibTeX entry should start with @")
        
        if not citation.endswith('}'):
            issues.append("BibTeX entry should end with }")
        
        if 'title=' not in citation:
            issues.append("Missing title field in BibTeX")
        
        return issues
    
    def export_citations(self, citations: List[str], format_type: str, filename: Optional[str] = None) -> str:
        """
        Export citations to a formatted string suitable for file output.
        
        Args:
            citations: List of citation strings
            format_type: Citation format
            filename: Optional filename for the export
            
        Returns:
            Formatted export string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        export_lines = [
            f"# Research Paper Citations - {format_type} Format",
            f"# Generated on: {timestamp}",
            f"# Total citations: {len(citations)}",
            "",
        ]
        
        if format_type.upper() == 'BIBTEX':
            # For BibTeX, just concatenate entries
            export_lines.extend(citations)
        else:
            # For other formats, number the citations
            for i, citation in enumerate(citations, 1):
                export_lines.append(f"{i}. {citation}")
                export_lines.append("")  # Add blank line between citations
        
        return "\n".join(export_lines)
    
    def get_citation_statistics(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        Get statistics about the papers for citation purposes.
        
        Args:
            papers: List of Paper objects
            
        Returns:
            Citation statistics
        """
        stats = {
            'total_papers': len(papers),
            'papers_with_authors': sum(1 for p in papers if p.authors),
            'papers_with_year': sum(1 for p in papers if p.year),
            'papers_with_venue': sum(1 for p in papers if p.venue),
            'papers_with_doi': sum(1 for p in papers if p.doi),
            'papers_with_url': sum(1 for p in papers if p.url),
            'year_range': self._get_year_range(papers),
            'most_common_venues': self._get_most_common_venues(papers, top_n=5)
        }
        
        return stats
    
    def _get_year_range(self, papers: List[Paper]) -> Dict[str, Optional[int]]:
        """Get the range of publication years"""
        years = [p.year for p in papers if p.year]
        
        if years:
            return {
                'earliest': min(years),
                'latest': max(years),
                'span': max(years) - min(years)
            }
        else:
            return {'earliest': None, 'latest': None, 'span': None}
    
    def _get_most_common_venues(self, papers: List[Paper], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the most common publication venues"""
        from collections import Counter
        
        venues = [p.venue for p in papers if p.venue]
        venue_counts = Counter(venues)
        
        return [
            {'venue': venue, 'count': count}
            for venue, count in venue_counts.most_common(top_n)
        ]
