"""
Coordinator Agent - Orchestrates the entire workflow and manages other agents.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import BaseAgent, AgentResult
from .paper_discovery import PaperDiscoveryAgent
from .paper_processing import PaperProcessingAgent
from .topic_classification import TopicClassificationAgent
from .summarization import SummarizationAgent
from .synthesis import SynthesisAgent
from .text_to_speech import TextToSpeechAgent
from .citation_manager import CitationManagerAgent
from utils.data_models import Paper, ProcessingResult, AudioResult

@dataclass
class WorkflowResult:
    """Result of a complete workflow execution"""
    success: bool
    papers: List[Paper]
    processing_results: Dict[str, ProcessingResult]
    audio_results: Dict[str, AudioResult] = None
    error: Optional[str] = None

class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent that orchestrates the entire multi-agent workflow.
    
    Responsibilities:
    - Initialize and manage all specialized agents
    - Coordinate workflows between agents
    - Handle user requests and route to appropriate agents
    - Manage data flow and state between processing steps
    """
    
    def __init__(self):
        super().__init__("Coordinator")
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        try:
            self.discovery_agent = PaperDiscoveryAgent()
            self.processing_agent = PaperProcessingAgent()
            self.classification_agent = TopicClassificationAgent()
            self.summarization_agent = SummarizationAgent()
            self.synthesis_agent = SynthesisAgent()
            self.tts_agent = TextToSpeechAgent()
            self.citation_agent = CitationManagerAgent()
            
            self.log_info("All agents initialized successfully")
            
        except Exception as e:
            self.log_error("Failed to initialize agents", e)
            raise
    
    async def process(self, input_data: Any, **kwargs) -> AgentResult:
        """Process input data through the appropriate workflow"""
        workflow_type = kwargs.get('workflow_type', 'search')
        
        if workflow_type == 'search':
            return await self.process_search_request(input_data)
        elif workflow_type == 'pdf_upload':
            return await self.process_pdf_uploads(input_data, kwargs.get('topics', []))
        elif workflow_type == 'doi':
            return await self.process_dois(input_data, kwargs.get('topics', []))
        elif workflow_type == 'url':
            return await self.process_urls(input_data, kwargs.get('topics', []))
        else:
            return AgentResult(success=False, error=f"Unknown workflow type: {workflow_type}")
    
    async def process_search_request(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a paper search request through the complete workflow.
        
        Args:
            search_params: Search parameters including query, filters, and topics
            
        Returns:
            Dict containing success status, papers, and processing results
        """
        try:
            self.log_info("Starting search workflow", query=search_params.get('query'))
            
            # Step 1: Discover papers
            discovery_result = await self.discovery_agent.process(search_params)
            if not discovery_result.success:
                return {'success': False, 'error': discovery_result.error}
            
            papers = discovery_result.data
            self.log_info(f"Discovered {len(papers)} papers")
            
            # Step 2: Process papers through the analysis pipeline
            processing_results = await self._process_papers_pipeline(papers, search_params['topics'])
            
            return {
                'success': True,
                'papers': papers,
                'processing_results': processing_results
            }
            
        except Exception as e:
            self.log_error("Search workflow failed", e)
            return {'success': False, 'error': str(e)}
    
    async def process_pdf_uploads(self, file_paths: List[str], topics: List[str]) -> Dict[str, Any]:
        """
        Process uploaded PDF files through the complete workflow.
        
        Args:
            file_paths: List of file paths to uploaded PDFs
            topics: List of research topics for classification
            
        Returns:
            Dict containing success status, papers, and processing results
        """
        try:
            self.log_info(f"Starting PDF upload workflow for {len(file_paths)} files")
            
            # Step 1: Process PDF files
            papers = []
            for file_path in file_paths:
                processing_result = await self.processing_agent.process(
                    {'type': 'pdf_file', 'path': file_path}
                )
                if processing_result.success:
                    papers.append(processing_result.data)
            
            self.log_info(f"Successfully processed {len(papers)} PDF files")
            
            # Step 2: Process papers through the analysis pipeline
            processing_results = await self._process_papers_pipeline(papers, topics)
            
            return {
                'success': True,
                'papers': papers,
                'processing_results': processing_results
            }
            
        except Exception as e:
            self.log_error("PDF upload workflow failed", e)
            return {'success': False, 'error': str(e)}
    
    async def process_dois(self, dois: List[str], topics: List[str]) -> Dict[str, Any]:
        """
        Process DOIs through the complete workflow.
        
        Args:
            dois: List of DOI identifiers
            topics: List of research topics for classification
            
        Returns:
            Dict containing success status, papers, and processing results
        """
        try:
            self.log_info(f"Starting DOI workflow for {len(dois)} DOIs")
            
            # Step 1: Process DOIs
            papers = []
            for doi in dois:
                processing_result = await self.processing_agent.process(
                    {'type': 'doi', 'doi': doi}
                )
                if processing_result.success:
                    papers.append(processing_result.data)
            
            self.log_info(f"Successfully processed {len(papers)} DOIs")
            
            # Step 2: Process papers through the analysis pipeline
            processing_results = await self._process_papers_pipeline(papers, topics)
            
            return {
                'success': True,
                'papers': papers,
                'processing_results': processing_results
            }
            
        except Exception as e:
            self.log_error("DOI workflow failed", e)
            return {'success': False, 'error': str(e)}
    
    async def process_urls(self, urls: List[str], topics: List[str]) -> Dict[str, Any]:
        """
        Process URLs through the complete workflow.
        
        Args:
            urls: List of paper URLs
            topics: List of research topics for classification
            
        Returns:
            Dict containing success status, papers, and processing results
        """
        try:
            self.log_info(f"Starting URL workflow for {len(urls)} URLs")
            
            # Step 1: Process URLs
            papers = []
            for url in urls:
                processing_result = await self.processing_agent.process(
                    {'type': 'url', 'url': url}
                )
                if processing_result.success:
                    papers.append(processing_result.data)
            
            self.log_info(f"Successfully processed {len(papers)} URLs")
            
            # Step 2: Process papers through the analysis pipeline
            processing_results = await self._process_papers_pipeline(papers, topics)
            
            return {
                'success': True,
                'papers': papers,
                'processing_results': processing_results
            }
            
        except Exception as e:
            self.log_error("URL workflow failed", e)
            return {'success': False, 'error': str(e)}
    
    async def _process_papers_pipeline(self, papers: List[Paper], topics: List[str]) -> Dict[str, ProcessingResult]:
        """
        Process papers through the analysis pipeline (classification, summarization).
        
        Args:
            papers: List of Paper objects
            topics: List of research topics
            
        Returns:
            Dict mapping paper IDs to ProcessingResult objects
        """
        processing_results = {}
        
        try:
            # Process papers concurrently
            tasks = []
            for paper in papers:
                task = self._process_single_paper(paper, topics)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for paper, result in zip(papers, results):
                if isinstance(result, Exception):
                    self.log_error(f"Failed to process paper {paper.id}", result)
                    # Create a basic processing result for failed papers
                    processing_results[paper.id] = ProcessingResult(
                        paper_id=paper.id,
                        topic="Unclassified",
                        summary="Failed to generate summary",
                        synthesis=None
                    )
                else:
                    processing_results[paper.id] = result
            
            self.log_info(f"Completed processing pipeline for {len(processing_results)} papers")
            
        except Exception as e:
            self.log_error("Processing pipeline failed", e)
        
        return processing_results
    
    async def _process_single_paper(self, paper: Paper, topics: List[str]) -> ProcessingResult:
        """
        Process a single paper through classification and summarization.
        
        Args:
            paper: Paper object to process
            topics: List of research topics
            
        Returns:
            ProcessingResult object
        """
        try:
            # Step 1: Topic classification
            classification_result = await self.classification_agent.process(
                {'paper': paper, 'topics': topics}
            )
            
            topic = "Unclassified"
            if classification_result.success:
                topic = classification_result.data
            
            # Step 2: Summarization
            summary_result = await self.summarization_agent.process(paper)
            
            summary = "Failed to generate summary"
            if summary_result.success:
                summary = summary_result.data
            
            return ProcessingResult(
                paper_id=paper.id,
                topic=topic,
                summary=summary,
                synthesis=None  # Will be filled later by synthesis step
            )
            
        except Exception as e:
            self.log_error(f"Failed to process single paper {paper.id}", e)
            raise
    
    async def generate_syntheses(self, processing_results: Dict[str, ProcessingResult], topics: List[str]) -> Dict[str, str]:
        """
        Generate cross-paper syntheses for topics with multiple papers.
        
        Args:
            processing_results: Dictionary of processing results
            topics: List of research topics
            
        Returns:
            Dict mapping paper IDs to synthesis text
        """
        try:
            self.log_info("Starting synthesis generation")
            synthesis_results = {}
            
            # Group papers by topic
            papers_by_topic = {}
            for paper_id, result in processing_results.items():
                topic = result.topic
                if topic not in papers_by_topic:
                    papers_by_topic[topic] = []
                papers_by_topic[topic].append((paper_id, result))
            
            # Generate synthesis for topics with multiple papers
            for topic, topic_papers in papers_by_topic.items():
                if len(topic_papers) > 1:  # Only synthesize if there are multiple papers
                    synthesis_input = {
                        'topic': topic,
                        'papers': [(pid, result.summary) for pid, result in topic_papers]
                    }
                    
                    synthesis_result = await self.synthesis_agent.process(synthesis_input)
                    
                    if synthesis_result.success:
                        synthesis_text = synthesis_result.data
                        # Assign the synthesis to all papers in this topic
                        for paper_id, _ in topic_papers:
                            synthesis_results[paper_id] = synthesis_text
            
            self.log_info(f"Generated syntheses for {len(synthesis_results)} papers")
            return synthesis_results
            
        except Exception as e:
            self.log_error("Synthesis generation failed", e)
            return {}
    
    async def generate_audio_summaries(self, processing_results: Dict[str, ProcessingResult]) -> Dict[str, AudioResult]:
        """
        Generate audio summaries for all processed papers.
        
        Args:
            processing_results: Dictionary of processing results
            
        Returns:
            Dict mapping paper IDs to AudioResult objects
        """
        try:
            self.log_info("Starting audio generation")
            audio_results = {}
            
            # Generate audio for each paper's summary
            tasks = []
            for paper_id, result in processing_results.items():
                if result.summary:
                    task = self._generate_single_audio(paper_id, result)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            paper_ids = list(processing_results.keys())
            for paper_id, result in zip(paper_ids, results):
                if not isinstance(result, Exception) and result:
                    audio_results[paper_id] = result
                else:
                    self.log_error(f"Failed to generate audio for paper {paper_id}", result if isinstance(result, Exception) else None)
            
            self.log_info(f"Generated audio for {len(audio_results)} papers")
            return audio_results
            
        except Exception as e:
            self.log_error("Audio generation failed", e)
            return {}
    
    async def _generate_single_audio(self, paper_id: str, result: ProcessingResult) -> AudioResult:
        """
        Generate audio for a single paper's summary.
        
        Args:
            paper_id: Paper identifier
            result: Processing result containing summary
            
        Returns:
            AudioResult object
        """
        try:
            # Combine summary and synthesis for richer audio content
            text_content = result.summary
            if result.synthesis:
                text_content += f"\n\nCross-paper insights: {result.synthesis}"
            
            audio_result = await self.tts_agent.process({
                'text': text_content,
                'paper_id': paper_id,
                'topic': result.topic
            })
            
            if audio_result.success:
                return audio_result.data
            else:
                self.log_error(f"TTS failed for paper {paper_id}: {audio_result.error}")
                return None
                
        except Exception as e:
            self.log_error(f"Failed to generate audio for paper {paper_id}", e)
            return None
    
    async def generate_citations(self, papers: List[Paper], format_type: str = "APA") -> List[str]:
        """
        Generate citations for papers.
        
        Args:
            papers: List of Paper objects
            format_type: Citation format (APA, MLA, Chicago, BibTeX)
            
        Returns:
            List of formatted citations
        """
        try:
            self.log_info(f"Generating {format_type} citations for {len(papers)} papers")
            
            citation_result = await self.citation_agent.process({
                'papers': papers,
                'format': format_type
            })
            
            if citation_result.success:
                return citation_result.data
            else:
                self.log_error(f"Citation generation failed: {citation_result.error}")
                return [f"Citation generation failed for paper: {paper.title}" for paper in papers]
                
        except Exception as e:
            self.log_error("Citation generation failed", e)
            return [f"Error generating citation for: {paper.title}" for paper in papers]
