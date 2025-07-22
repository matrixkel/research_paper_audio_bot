# Multi-Agent Research Paper Analysis System

## Overview

This is a sophisticated research paper analysis system built with Streamlit that uses a multi-agent architecture to discover, process, analyze, and synthesize academic research papers. The system integrates with multiple academic APIs and uses AI-powered agents to provide comprehensive paper analysis, summarization, and cross-paper synthesis capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - Python-based web framework for rapid UI development
- **Layout**: Wide layout with expandable sidebar for configuration
- **State Management**: Streamlit session state for maintaining application state across user interactions
- **User Interface**: Simple, intuitive interface for uploading papers, configuring settings, and viewing results

### Backend Architecture
- **Multi-Agent System**: 8 specialized agents working in coordination
- **Asynchronous Processing**: Built on asyncio for concurrent operations
- **API Integration**: Multiple external APIs for paper discovery and AI processing
- **Modular Design**: Clear separation of concerns with dedicated agents for specific tasks

### Agent Architecture
The system uses a coordinated multi-agent approach:

1. **Coordinator Agent**: Central orchestrator managing workflow and inter-agent communication
2. **Paper Discovery Agent**: Searches academic databases (Semantic Scholar, ArXiv)
3. **Paper Processing Agent**: Extracts text from PDFs, URLs, and DOIs
4. **Topic Classification Agent**: Uses semantic embeddings for automatic categorization
5. **Summarization Agent**: Generates AI-powered summaries using Groq API
6. **Synthesis Agent**: Creates cross-paper insights and analyses
7. **Text-to-Speech Agent**: Converts summaries to audio podcasts
8. **Citation Manager Agent**: Handles citation formatting in multiple styles

## Key Components

### Core Technologies
- **AI/LLM**: Groq API with Llama models for natural language processing
- **Document Processing**: PyMuPDF for PDF text extraction
- **Semantic Analysis**: Sentence Transformers for embeddings and similarity
- **Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS)
- **Web Scraping**: Trafilatura for content extraction from URLs

### Data Models
- **Paper**: Core data structure containing metadata, content, and processing state
- **ProcessingResult**: Results from paper analysis and summarization
- **AudioResult**: Audio file generation results and metadata
- **AgentResult**: Standardized result format for agent operations

### Configuration Management
- Centralized configuration system managing API keys, storage paths, and processing parameters
- Environment variable support for sensitive configuration
- Configurable rate limiting and processing constraints

## Data Flow

### Paper Discovery and Processing
1. User provides input (PDF upload, DOI, URL, or search query)
2. Coordinator routes request to appropriate agent
3. Discovery Agent searches academic databases if needed
4. Processing Agent extracts text and metadata
5. Classification Agent categorizes papers by topic
6. Results stored in session state for further processing

### Analysis and Synthesis
1. Summarization Agent generates individual paper summaries using Groq API
2. Papers grouped by topic using semantic similarity
3. Synthesis Agent creates cross-paper analyses for topics with multiple papers
4. Citation Manager generates formatted references
5. TTS Agent converts text to audio on demand

### State Management
- Session state maintains papers, processing results, and configuration
- Asynchronous processing with proper error handling and logging
- Storage directory structure for papers, audio files, and temporary data

## External Dependencies

### API Services
- **Groq API**: Primary LLM service for summarization and synthesis
- **Semantic Scholar API**: Academic paper search and metadata
- **ArXiv API**: Preprint paper search
- **CrossRef API**: DOI resolution and metadata

### Required Libraries
- **Core**: streamlit, asyncio, aiohttp
- **AI/ML**: groq, sentence-transformers
- **Document Processing**: PyMuPDF (fitz), trafilatura
- **Audio**: pyttsx3, gTTS
- **Data**: pandas, numpy

### Storage Requirements
- Local file system for paper storage, audio files, and temporary data
- Configurable storage paths with automatic directory creation
- Audio cleanup with configurable retention period

## Deployment Strategy

### Local Development
- Python 3.8+ environment with pip dependencies
- Streamlit server for local development and testing
- Environment variables for API key configuration
- Directory structure automatically created on startup

### Production Considerations
- Scalable to handle multiple concurrent users
- Rate limiting implemented for all external APIs
- Error handling and logging throughout the system
- Configurable resource limits for processing

### Configuration
- API keys managed through environment variables or UI input
- Adjustable processing limits and timeouts
- Configurable storage locations and cleanup policies
- Rate limiting settings for each external service

### Performance Optimization
- Asynchronous processing for I/O operations
- Concurrent request handling with configurable limits
- Caching of embeddings and processed results
- Lightweight sentence transformer model for fast classification