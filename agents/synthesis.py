"""
Synthesis Agent - Generates cross-paper topic syntheses using Groq API.
"""

import asyncio
from typing import List, Tuple, Dict, Any
from groq import AsyncGroq

from .base import BaseAgent, AgentResult
from utils.config import Config

class SynthesisAgent(BaseAgent):
    """
    Agent responsible for generating cross-paper syntheses that identify patterns,
    trends, and insights across multiple papers within the same topic.
    
    Creates coherent narratives that:
    - Identify common themes and patterns
    - Highlight contradictions or debates
    - Synthesize collective insights
    - Identify research gaps and future directions
    """
    
    def __init__(self):
        super().__init__("Synthesis")
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
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Generate a synthesis across multiple papers on the same topic.
        
        Args:
            input_data: Dictionary containing:
                - topic: The topic name
                - papers: List of (paper_id, summary) tuples
                
        Returns:
            AgentResult containing the synthesis text
        """
        return await self.safe_execute("topic_synthesis", self._synthesize_topic, input_data)
    
    async def _synthesize_topic(self, input_data: Dict[str, Any]) -> str:
        """
        Generate a synthesis for papers within a topic.
        
        Args:
            input_data: Contains topic name and paper summaries
            
        Returns:
            Synthesis text combining insights from multiple papers
        """
        if not self.client:
            raise ValueError("Groq client not initialized. Please check API key configuration.")
        
        topic = input_data['topic']
        papers = input_data['papers']  # List of (paper_id, summary) tuples
        
        self.log_info(f"Generating synthesis for topic '{topic}' with {len(papers)} papers")
        
        if len(papers) < 2:
            return self._generate_single_paper_note(topic, papers[0][1] if papers else "")
        
        # Prepare the synthesis prompt
        prompt = self._create_synthesis_prompt(topic, papers)
        
        try:
            # Call Groq API for synthesis
            response = await self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Use larger model for complex synthesis
                messages=[
                    {
                        "role": "system",
                        "content": self._get_synthesis_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,  # Moderate temperature for creative synthesis
                max_tokens=1500,  # Longer output for comprehensive synthesis
                top_p=0.9
            )
            
            synthesis = response.choices[0].message.content.strip()
            
            self.log_info(f"Successfully generated synthesis for '{topic}' ({len(synthesis)} characters)")
            return synthesis
            
        except Exception as e:
            self.log_error(f"Failed to generate synthesis via Groq API", e)
            # Fall back to template-based synthesis
            return self._generate_fallback_synthesis(topic, papers)
    
    def _create_synthesis_prompt(self, topic: str, papers: List[Tuple[str, str]]) -> str:
        """
        Create a comprehensive prompt for topic synthesis.
        
        Args:
            topic: The research topic
            papers: List of (paper_id, summary) tuples
            
        Returns:
            Synthesis prompt
        """
        # Prepare paper summaries for the prompt
        paper_summaries = []
        for i, (paper_id, summary) in enumerate(papers, 1):
            paper_summaries.append(f"Paper {i}:\n{summary}")
        
        summaries_text = "\n\n".join(paper_summaries)
        
        prompt = f"""I have {len(papers)} research papers all related to the topic "{topic}". Please create a comprehensive synthesis that analyzes these papers collectively.

Here are the individual paper summaries:

{summaries_text}

Please generate a synthesis that addresses the following aspects:

1. **Common Themes & Patterns**: What recurring themes, methodologies, or findings appear across multiple papers?

2. **Convergent Insights**: Where do the papers agree or reinforce each other's findings?

3. **Divergent Views & Debates**: What contradictions, disagreements, or different perspectives exist between the papers?

4. **Collective Contributions**: What is the combined contribution of these papers to our understanding of {topic}?

5. **Research Landscape**: How do these papers collectively shape or advance the field?

6. **Gaps & Future Directions**: What research gaps become apparent when looking at these papers together? What future research directions are suggested?

7. **Methodological Insights**: What can we learn about research approaches and methodologies from comparing these papers?

Please create a coherent narrative that weaves together insights from all papers while maintaining academic rigor. The synthesis should be valuable for researchers wanting to understand the current state and trends in {topic} research.
"""
        
        return prompt
    
    def _get_synthesis_system_prompt(self) -> str:
        """
        Get the system prompt that defines the AI's role for synthesis.
        
        Returns:
            System prompt string
        """
        return """You are an expert research analyst specializing in synthesizing insights across multiple academic papers. Your role is to:

- Identify patterns, trends, and connections across research papers
- Create coherent narratives that combine insights from multiple sources
- Highlight both convergent and divergent findings
- Provide meta-level insights about the research landscape
- Maintain academic rigor while being accessible
- Focus on analysis and synthesis rather than just summarization
- Use clear structure and organization
- Avoid speculation and stay grounded in the provided evidence

Your syntheses should help researchers understand not just individual papers, but the broader picture and collective insights that emerge when multiple papers are considered together. Think of yourself as creating a "research landscape map" that shows how different papers relate to each other and contribute to the field."""
    
    def _generate_single_paper_note(self, topic: str, summary: str) -> str:
        """
        Generate a note when only one paper is available for a topic.
        
        Args:
            topic: The topic name
            summary: The single paper's summary
            
        Returns:
            Note explaining single paper status
        """
        return f"""**Single Paper Analysis for Topic: {topic}**

This topic currently contains only one paper, so a cross-paper synthesis is not applicable. However, here are the key insights from the available research:

{summary}

**Note for Future Research:**
As more papers are added to this topic, a comprehensive synthesis comparing methodologies, findings, and perspectives across multiple studies will become available. This will provide deeper insights into trends, debates, and the overall research landscape in {topic}.

**Current Research Gap:**
The limited number of papers in this topic suggests either a specialized/niche area or an emerging field that would benefit from additional research contributions."""
    
    def _generate_fallback_synthesis(self, topic: str, papers: List[Tuple[str, str]]) -> str:
        """
        Generate a basic fallback synthesis when Groq API is unavailable.
        
        Args:
            topic: The topic name
            papers: List of (paper_id, summary) tuples
            
        Returns:
            Basic template-based synthesis
        """
        self.log_info("Generating fallback synthesis")
        
        synthesis_parts = []
        
        synthesis_parts.append(f"**Cross-Paper Synthesis: {topic}**")
        synthesis_parts.append(f"*Analysis of {len(papers)} papers*")
        
        synthesis_parts.append(f"\n**Papers Included in This Synthesis:**")
        for i, (paper_id, summary) in enumerate(papers, 1):
            # Extract title from summary if possible
            title_line = summary.split('\n')[0] if summary else f"Paper {i}"
            if title_line.startswith("Title:"):
                title_line = title_line.replace("Title:", "").strip()
            synthesis_parts.append(f"{i}. {title_line}")
        
        synthesis_parts.append(f"\n**Key Insights:**")
        
        # Extract key points from summaries
        all_text = " ".join([summary for _, summary in papers])
        
        # Look for common terms and themes (basic analysis)
        common_terms = self._extract_common_terms(all_text)
        if common_terms:
            synthesis_parts.append(f"Common research themes include: {', '.join(common_terms[:5])}")
        
        synthesis_parts.append(f"\n**Individual Paper Contributions:**")
        for i, (paper_id, summary) in enumerate(papers, 1):
            # Extract first few sentences as key contribution
            sentences = summary.split('.')[:2]
            contribution = '.'.join(sentences) + '.' if sentences else "Key insights from this research."
            synthesis_parts.append(f"\nPaper {i}: {contribution}")
        
        synthesis_parts.append(f"\n**Synthesis Note:**")
        synthesis_parts.append("This is a basic synthesis generated without advanced AI analysis. " +
                             "For deeper insights into patterns, contradictions, and collective " +
                             "contributions, full API access is required.")
        
        return '\n'.join(synthesis_parts)
    
    def _extract_common_terms(self, text: str, min_length: int = 4) -> List[str]:
        """
        Extract common terms from combined text (basic keyword extraction).
        
        Args:
            text: Combined text from all papers
            min_length: Minimum word length to consider
            
        Returns:
            List of common terms
        """
        import re
        from collections import Counter
        
        # Simple term extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'research', 'study',
            'paper', 'analysis', 'method', 'approach', 'results', 'findings'
        }
        
        filtered_words = [word for word in words 
                         if len(word) >= min_length and word not in stop_words]
        
        # Count frequency and return most common
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(10) if count > 1]
    
    async def batch_synthesize(self, topic_papers: Dict[str, List[Tuple[str, str]]]) -> Dict[str, str]:
        """
        Generate syntheses for multiple topics efficiently.
        
        Args:
            topic_papers: Dictionary mapping topic names to lists of (paper_id, summary) tuples
            
        Returns:
            Dictionary mapping topic names to synthesis texts
        """
        self.log_info(f"Starting batch synthesis for {len(topic_papers)} topics")
        
        # Filter topics that have multiple papers (synthesis only makes sense with 2+ papers)
        synthesizable_topics = {
            topic: papers for topic, papers in topic_papers.items()
            if len(papers) >= 2
        }
        
        self.log_info(f"Synthesizing {len(synthesizable_topics)} topics with multiple papers")
        
        # Process topics concurrently (with rate limiting)
        semaphore = asyncio.Semaphore(2)  # Limit concurrent syntheses
        
        async def synthesize_with_semaphore(topic, papers):
            async with semaphore:
                result = await self.process({'topic': topic, 'papers': papers})
                return topic, result.data if result.success else None
        
        tasks = [
            synthesize_with_semaphore(topic, papers)
            for topic, papers in synthesizable_topics.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        syntheses = {}
        for result in results:
            if isinstance(result, Exception):
                self.log_error("Batch synthesis task failed", result)
                continue
            
            topic, synthesis = result
            if synthesis:
                syntheses[topic] = synthesis
        
        self.log_info(f"Batch synthesis completed: {len(syntheses)} successful")
        return syntheses
    
    def analyze_synthesis_coverage(self, papers: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze how well a synthesis would cover the provided papers.
        
        Args:
            papers: List of (paper_id, summary) tuples
            
        Returns:
            Dictionary with coverage analysis
        """
        analysis = {
            'total_papers': len(papers),
            'synthesizable': len(papers) >= 2,
            'estimated_synthesis_length': min(1500, len(papers) * 200),  # Rough estimate
            'content_diversity': self._estimate_content_diversity(papers),
            'synthesis_complexity': 'High' if len(papers) > 5 else 'Medium' if len(papers) > 2 else 'Low'
        }
        
        return analysis
    
    def _estimate_content_diversity(self, papers: List[Tuple[str, str]]) -> str:
        """
        Estimate content diversity across papers (basic heuristic).
        
        Args:
            papers: List of (paper_id, summary) tuples
            
        Returns:
            Diversity estimate ('High', 'Medium', 'Low')
        """
        if len(papers) < 2:
            return 'N/A'
        
        # Simple heuristic: compare summary lengths and word overlap
        summaries = [summary for _, summary in papers]
        
        # Check length diversity
        lengths = [len(summary.split()) for summary in summaries]
        length_variance = max(lengths) - min(lengths) if lengths else 0
        
        # Check word overlap (very basic)
        all_words = set()
        for summary in summaries:
            all_words.update(summary.lower().split())
        
        unique_words_per_paper = []
        for summary in summaries:
            paper_words = set(summary.lower().split())
            unique_ratio = len(paper_words) / len(all_words) if all_words else 0
            unique_words_per_paper.append(unique_ratio)
        
        avg_uniqueness = sum(unique_words_per_paper) / len(unique_words_per_paper)
        
        if avg_uniqueness > 0.7 or length_variance > 100:
            return 'High'
        elif avg_uniqueness > 0.4 or length_variance > 50:
            return 'Medium'
        else:
            return 'Low'
