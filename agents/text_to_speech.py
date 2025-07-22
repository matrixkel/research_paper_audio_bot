"""
Text-to-Speech Agent - Converts text summaries to audio using TTS engines.
"""

import asyncio
import os
import uuid
from typing import Dict, Any, Optional
import tempfile

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from .base import BaseAgent, AgentResult
from utils.data_models import AudioResult

class TextToSpeechAgent(BaseAgent):
    """
    Agent responsible for converting text summaries into audio files.
    
    Supports multiple TTS engines:
    - pyttsx3 (offline, fast)
    - Google Text-to-Speech (online, high quality)
    
    Generates audio files for paper summaries and syntheses.
    """
    
    def __init__(self):
        super().__init__("TextToSpeech")
        self.tts_engine = None
        self._initialize_tts()
        
        # Ensure audio storage directory exists
        self.audio_dir = "storage/audio"
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def _initialize_tts(self):
        """Initialize the preferred TTS engine"""
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self._configure_pyttsx3()
                self.preferred_engine = "pyttsx3"
                self.log_info("pyttsx3 TTS engine initialized successfully")
            except Exception as e:
                self.log_warning("Failed to initialize pyttsx3", error=str(e))
                self.tts_engine = None
                self.preferred_engine = "gtts" if GTTS_AVAILABLE else None
        else:
            self.preferred_engine = "gtts" if GTTS_AVAILABLE else None
        
        if not self.preferred_engine:
            self.log_error("No TTS engines available. Install pyttsx3 or gtts.")
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 settings for better audio quality"""
        if self.tts_engine:
            try:
                # Set speech rate (words per minute)
                self.tts_engine.setProperty('rate', 180)
                
                # Set volume (0.0 to 1.0)
                self.tts_engine.setProperty('volume', 0.9)
                
                # Try to set a clear voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Prefer female voices or first available voice
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                    else:
                        # Use first available voice
                        self.tts_engine.setProperty('voice', voices[0].id)
                
            except Exception as e:
                self.log_warning("Failed to configure pyttsx3 settings", error=str(e))
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Convert text to speech and save as audio file.
        
        Args:
            input_data: Dictionary containing:
                - text: Text content to convert
                - paper_id: Paper identifier for filename
                - topic: Topic classification for organization
                
        Returns:
            AgentResult containing AudioResult with file path and metadata
        """
        return await self.safe_execute("text_to_speech_conversion", self._convert_to_audio, input_data)
    
    async def _convert_to_audio(self, input_data: Dict[str, Any]) -> AudioResult:
        """
        Main text-to-speech conversion method.
        
        Args:
            input_data: Contains text, paper_id, and topic
            
        Returns:
            AudioResult with file path and metadata
        """
        text = input_data['text']
        paper_id = input_data['paper_id']
        topic = input_data.get('topic', 'general')
        
        if not text or not text.strip():
            raise ValueError("No text provided for audio conversion")
        
        self.log_info(f"Converting text to audio for paper {paper_id}")
        
        # Prepare audio content
        audio_text = self._prepare_audio_text(text, topic)
        
        # Generate unique filename
        audio_filename = f"{paper_id}_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        # Convert to audio using available engine
        if self.preferred_engine == "pyttsx3":
            await self._convert_with_pyttsx3(audio_text, audio_path)
        elif self.preferred_engine == "gtts":
            await self._convert_with_gtts(audio_text, audio_path)
        else:
            raise RuntimeError("No TTS engine available")
        
        # Verify file was created
        if not os.path.exists(audio_path):
            raise RuntimeError("Audio file was not created successfully")
        
        # Get file size
        file_size = os.path.getsize(audio_path)
        
        audio_result = AudioResult(
            paper_id=paper_id,
            file_path=audio_path,
            filename=audio_filename,
            duration_seconds=self._estimate_duration(audio_text),
            file_size_bytes=file_size,
            format="mp3",
            engine_used=self.preferred_engine
        )
        
        self.log_info(f"Successfully created audio file: {audio_filename} ({file_size} bytes)")
        return audio_result
    
    def _prepare_audio_text(self, text: str, topic: str) -> str:
        """
        Prepare text for audio conversion by adding intro and formatting.
        
        Args:
            text: Original text content
            topic: Topic classification
            
        Returns:
            Formatted text ready for TTS
        """
        # Add introduction
        intro = f"Research paper summary for topic: {topic}. "
        
        # Clean up text for better speech
        cleaned_text = self._clean_text_for_speech(text)
        
        # Add conclusion
        outro = " This concludes the research paper summary."
        
        return intro + cleaned_text + outro
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean text to make it more suitable for text-to-speech.
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove or replace markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markers
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic markers
        text = re.sub(r'#{1,6}\s*', '', text)         # Remove headers
        
        # Replace some symbols with words for better pronunciation
        replacements = {
            '&': ' and ',
            '%': ' percent ',
            '@': ' at ',
            '#': ' number ',
            '$': ' dollar ',
            '€': ' euro ',
            '£': ' pound ',
            '+': ' plus ',
            '=': ' equals ',
            '<': ' less than ',
            '>': ' greater than ',
            '–': ' to ',  # en dash
            '—': ' to ',  # em dash
        }
        
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
        
        # Improve pronunciation of common academic terms
        academic_replacements = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'et al.': 'and others',
            'vs.': 'versus',
            'etc.': 'and so on',
            'DOI': 'D O I',
            'URL': 'U R L',
            'API': 'A P I',
            'AI': 'A I',
            'ML': 'machine learning',
            'NLP': 'natural language processing',
            'PhD': 'P H D',
        }
        
        for abbrev, expansion in academic_replacements.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def _convert_with_pyttsx3(self, text: str, output_path: str):
        """
        Convert text to audio using pyttsx3 (offline TTS).
        
        Args:
            text: Text to convert
            output_path: Output file path
        """
        if not self.tts_engine:
            raise RuntimeError("pyttsx3 engine not initialized")
        
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, self._pyttsx3_save_to_file, text, output_path
        )
    
    def _pyttsx3_save_to_file(self, text: str, output_path: str):
        """
        Save pyttsx3 output to file (runs in thread pool).
        
        Args:
            text: Text to convert
            output_path: Output file path
        """
        try:
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
        except Exception as e:
            self.log_error("pyttsx3 conversion failed", e)
            raise
    
    async def _convert_with_gtts(self, text: str, output_path: str):
        """
        Convert text to audio using Google Text-to-Speech.
        
        Args:
            text: Text to convert
            output_path: Output file path
        """
        if not GTTS_AVAILABLE:
            raise RuntimeError("gTTS not available")
        
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, self._gtts_save_to_file, text, output_path
        )
    
    def _gtts_save_to_file(self, text: str, output_path: str):
        """
        Save gTTS output to file (runs in thread pool).
        
        Args:
            text: Text to convert
            output_path: Output file path
        """
        try:
            # Split text if too long for gTTS (max ~5000 chars)
            if len(text) > 4000:
                text = text[:4000] + "... Content truncated for audio generation."
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            
        except Exception as e:
            self.log_error("gTTS conversion failed", e)
            raise
    
    def _estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration based on text length.
        
        Args:
            text: Text content
            
        Returns:
            Estimated duration in seconds
        """
        # Rough estimate: average reading speed is ~150-200 words per minute
        words = len(text.split())
        estimated_minutes = words / 170  # Conservative estimate
        return estimated_minutes * 60
    
    async def batch_convert(self, text_items: list, max_concurrent: int = 3) -> Dict[str, AudioResult]:
        """
        Convert multiple texts to audio efficiently.
        
        Args:
            text_items: List of input data dictionaries
            max_concurrent: Maximum concurrent conversions
            
        Returns:
            Dictionary mapping paper IDs to AudioResult objects
        """
        self.log_info(f"Starting batch audio conversion for {len(text_items)} items")
        
        # Use semaphore to limit concurrent conversions
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def convert_with_semaphore(item):
            async with semaphore:
                result = await self.process(item)
                return item['paper_id'], result.data if result.success else None
        
        # Process items concurrently
        tasks = [convert_with_semaphore(item) for item in text_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        audio_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.log_error("Batch conversion task failed", result)
                continue
            
            paper_id, audio_result = result
            if audio_result:
                audio_results[paper_id] = audio_result
        
        self.log_info(f"Batch conversion completed: {len(audio_results)}/{len(text_items)} successful")
        return audio_results
    
    def get_available_engines(self) -> Dict[str, bool]:
        """
        Get status of available TTS engines.
        
        Returns:
            Dictionary with engine availability status
        """
        return {
            'pyttsx3': PYTTSX3_AVAILABLE and self.tts_engine is not None,
            'gtts': GTTS_AVAILABLE,
            'preferred': self.preferred_engine
        }
    
    def cleanup_old_files(self, max_age_days: int = 7):
        """
        Clean up old audio files to save storage space.
        
        Args:
            max_age_days: Maximum age of files to keep
        """
        import time
        
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            removed_count = 0
            for filename in os.listdir(self.audio_dir):
                if filename.endswith('.mp3'):
                    file_path = os.path.join(self.audio_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        removed_count += 1
            
            if removed_count > 0:
                self.log_info(f"Cleaned up {removed_count} old audio files")
        
        except Exception as e:
            self.log_error("Failed to cleanup old audio files", e)
    
    def get_audio_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generated audio files.
        
        Returns:
            Dictionary with audio file statistics
        """
        try:
            audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.mp3')]
            
            if not audio_files:
                return {
                    'total_files': 0,
                    'total_size_mb': 0,
                    'average_size_mb': 0
                }
            
            total_size = sum(
                os.path.getsize(os.path.join(self.audio_dir, f)) 
                for f in audio_files
            )
            
            return {
                'total_files': len(audio_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'average_size_mb': round(total_size / (1024 * 1024) / len(audio_files), 2)
            }
        
        except Exception as e:
            self.log_error("Failed to get audio stats", e)
            return {'error': str(e)}
