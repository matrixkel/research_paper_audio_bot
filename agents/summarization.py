"""
Summarization Agent - Generates individual paper summaries using Groq API.
"""

import asyncio
from typing import Optional, Dict, Any
from groq import AsyncGroq

from .base import BaseAgent, AgentResult
from utils.data_models import Paper
from utils.config import Config

class SummarizationAgent(BaseAgent):
    """
    Agent responsible for generating comprehensive summaries of individual research papers.
    
    Uses Groq API to create structured summaries that include:
    - Key findings and contributions
    - Methodology overview
    - Significance and implications
    - Limitations and future work
    """
    
    def __init__(self):
        super().__init__("Summarization")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Groq client"""
        try:
            if Config.GROQ_API_KEY:
                self.client = AsyncGroq(api_key=Config.GROQ_API_KEY)
                self.log_info("Groq client initialized successfully")
            else:
                self.log_error("Groq API key not configured")
        except Exception as e:
            self.log_error("Failed to initialize Groq client", e)
    
    async def process(self, input_data: Paper, **kwargs) -> AgentResult:
        """
        Generate a summary for a research paper.
        
        Args:
            input_data: Paper object to summarize
            
        Returns:
            AgentResult containing the generated summary
        """
        return await self.safe_execute("paper_summarization", self._summarize_paper, input_data)
    
    async def _summarize_paper(self, paper: Paper) -> str:
        """
        Generate a comprehensive summary of a research paper.
        
        Args:
            paper: Paper object containing title, abstract, and other metadata
            
        Returns:
            Structured summary string
        """
        if not self.client:
            raise ValueError("Groq client not initialized. Please check API key configuration.")
        
        self.log_info(f"Generating summary for paper: {paper.title[:50]}...")
        
        # Prepare content for summarization
        content_to_summarize = self._prepare_content_for_summarization(paper)
        
        # Create the summarization prompt
        prompt = self._create_summarization_prompt(paper, content_to_summarize)
        
        try:
            # Call Groq API for summarization
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Fast model for summarization
                messages=[
                    {
                        "role": "system",
                        "content": self._get_summarization_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=1000,  # Reasonable length for summaries
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            self.log_info(f"Successfully generated summary ({len(summary)} characters)")
            return summary
            
        except Exception as e:
            self.log_error(f"Failed to generate summary via Groq API", e)
            # Fall back to template-based summary
            return self._generate_fallback_summary(paper)
    
    def _prepare_content_for_summarization(self, paper: Paper) -> str:
        """
        Prepare paper content for summarization by combining available text.
        
        Args:
            paper: Paper object
            
        Returns:
            Combined text content for summarization
        """
        content_parts = []
        
        # Add title
        if paper.title:
            content_parts.append(f"Title: {paper.title}")
        
        # Add authors and publication info
        if paper.authors:
            authors_str = ", ".join(paper.authors[:5])  # Limit to first 5 authors
            if len(paper.authors) > 5:
                authors_str += " et al."
            content_parts.append(f"Authors: {authors_str}")
        
        if paper.year:
            content_parts.append(f"Year: {paper.year}")
        
        if paper.venue:
            content_parts.append(f"Published in: {paper.venue}")
        
        # Add abstract
        if paper.abstract:
            content_parts.append(f"Abstract: {paper.abstract}")
        
        # Add full text if available (limit to reasonable length)
        if hasattr(paper, 'full_text') and paper.full_text:
            # Use first 2000 words to stay within token limits
            full_text_words = paper.full_text.split()[:2000]
            full_text_excerpt = ' '.join(full_text_words)
            content_parts.append(f"Full text excerpt: {full_text_excerpt}")
        
        return '\n\n'.join(content_parts)
    
    def _create_summarization_prompt(self, paper: Paper, content: str) -> str:
        """
        Create a detailed prompt for paper summarization.
        
        Args:
            paper: Paper object
            content: Prepared content text
            
        Returns:
            Summarization prompt
        """
        prompt = f"""Please analyze and summarize the following research paper:

{content}

Generate a comprehensive summary that includes:

1. **Key Findings & Contributions**: What are the main discoveries or contributions of this research?

2. **Methodology**: What research methods, approaches, or techniques were used?

3. **Significance & Impact**: Why is this research important? What are its implications for the field?

4. **Limitations & Future Work**: What are the acknowledged limitations? What future research directions are suggested?

5. **Context & Background**: How does this work relate to existing research in the field?

Please provide a well-structured summary that would be valuable for researchers wanting to quickly understand the paper's content and significance. Focus on being accurate, informative, and concise while covering all important aspects.
"""
        
        return prompt
    
    def _get_summarization_system_prompt(self) -> str:
        """
        Get the system prompt for summarization that defines the AI's role.
        
        Returns:
            System prompt string
        """
        return """You are an expert academic research analyst specializing in creating comprehensive, accurate summaries of research papers. Your summaries should:

- Be informative and well-structured
- Capture the essential contributions and findings
- Explain complex concepts clearly
- Maintain academic rigor while being accessible
- Focus on factual information from the paper
- Avoid speculation or adding information not present in the source
- Use clear section headers to organize the information
- Be concise but comprehensive, typically 300-800 words

Your goal is to help researchers quickly understand the key aspects of a paper without having to read the entire document."""
    
    def _generate_fallback_summary(self, paper: Paper) -> str:
        """
        Generate a basic fallback summary when Groq API is unavailable.
        
        Args:
            paper: Paper object
            
        Returns:
            Basic template-based summary
        """
        self.log_info("Generating fallback summary")
        
        summary_parts = []
        
        # Basic paper information
        summary_parts.append(f"**Research Paper Summary**")
        summary_parts.append(f"Title: {paper.title}")
        
        if paper.authors:
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            summary_parts.append(f"Authors: {authors_str}")
        
        if paper.year:
            summary_parts.append(f"Publication Year: {paper.year}")
        
        if paper.venue:
            summary_parts.append(f"Published in: {paper.venue}")
        
        # Abstract-based summary
        if paper.abstract:
            summary_parts.append(f"\n**Abstract Summary:**")
            # Truncate abstract if too long
            abstract_text = paper.abstract
            if len(abstract_text) > 500:
                abstract_text = abstract_text[:500] + "..."
            summary_parts.append(abstract_text)
        
        # Additional information if available
        if paper.citation_count and paper.citation_count > 0:
            summary_parts.append(f"\n**Citation Count:** {paper.citation_count}")
        
        if paper.doi:
            summary_parts.append(f"**DOI:** {paper.doi}")
        
        summary_parts.append(f"\n*Note: This is a basic summary. Full analysis requires API access.*")
        
        return '\n'.join(summary_parts)
    
    async def batch_summarize(self, papers: list, max_concurrent: int = 3) -> Dict[str, str]:
        """
        Summarize multiple papers efficiently with concurrency control.
        
        Args:
            papers: List of Paper objects
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Dictionary mapping paper IDs to summaries
        """
        self.log_info(f"Starting batch summarization of {len(papers)} papers")
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def summarize_with_semaphore(paper):
            async with semaphore:
                result = await self.process(paper)
                return paper.id, result.data if result.success else None
        
        # Process papers concurrently
        tasks = [summarize_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        summaries = {}
        for result in results:
            if isinstance(result, Exception):
                self.log_error("Batch summarization task failed", result)
                continue
            
            paper_id, summary = result
            if summary:
                summaries[paper_id] = summary
        
        self.log_info(f"Batch summarization completed: {len(summaries)}/{len(papers)} successful")
        return summaries
    
    def validate_summary_quality(self, summary: str, paper: Paper) -> Dict[str, Any]:
        """
        Validate the quality of a generated summary.
        
        Args:
            summary: Generated summary text
            paper: Original paper object
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'length_appropriate': 100 <= len(summary) <= 2000,
            'has_structure': any(marker in summary for marker in ['**', '###', '1.', '2.']),
            'mentions_title': paper.title.lower() in summary.lower() if paper.title else False,
            'not_too_short': len(summary) >= 100,
            'not_too_long': len(summary) <= 2000,
            'word_count': len(summary.split()),
            'character_count': len(summary)
        }
        
        # Calculate overall quality score
        positive_indicators = sum([
            quality_metrics['length_appropriate'],
            quality_metrics['has_structure'],
            quality_metrics['not_too_short'],
            quality_metrics['not_too_long']
        ])
        
        quality_metrics['quality_score'] = positive_indicators / 4.0
        
        return quality_metrics
