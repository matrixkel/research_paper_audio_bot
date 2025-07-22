# Multi-Agent Research Paper Analysis System

A sophisticated research paper analysis system built with Streamlit and powered by Groq API. This system uses a multi-agent architecture to discover, process, analyze, and synthesize research papers, generating both text summaries and audio podcasts.

## 🚀 Features

### Core Capabilities
- **Multi-Source Paper Discovery**: Search across Semantic Scholar and ArXiv APIs
- **Flexible Input Methods**: Upload PDFs, enter DOIs, or provide URLs
- **Intelligent Topic Classification**: Automatic categorization using semantic similarity
- **AI-Powered Summarization**: Generate comprehensive summaries using Groq API
- **Cross-Paper Synthesis**: Create insights across multiple papers in the same topic
- **Audio Generation**: Convert summaries to high-quality audio podcasts
- **Citation Management**: Generate citations in APA, MLA, Chicago, and BibTeX formats

### Multi-Agent Architecture
The system employs 8 specialized agents working together:

1. **Coordinator Agent**: Orchestrates workflows and manages inter-agent communication
2. **Paper Discovery Agent**: Searches and retrieves papers from academic databases
3. **Paper Processing Agent**: Extracts text and metadata from various sources
4. **Topic Classification Agent**: Categorizes papers using semantic embeddings
5. **Summarization Agent**: Generates individual paper summaries with Groq API
6. **Synthesis Agent**: Creates cross-paper topic syntheses
7. **Text-to-Speech Agent**: Converts text to audio using multiple TTS engines
8. **Citation Manager Agent**: Handles citation formatting and reference management

## 🛠️ Technology Stack

### Core Technologies
- **Frontend**: Streamlit (Python web framework)
- **AI/LLM**: Groq API (Llama models)
- **Document Processing**: PyMuPDF (PDF text extraction)
- **Semantic Search**: Sentence Transformers (embeddings)
- **Text-to-Speech**: pyttsx3, gTTS
- **Web Scraping**: Trafilatura

### APIs & Data Sources
- **Groq API**: LLM operations (summarization, synthesis)
- **Semantic Scholar API**: Academic paper search
- **ArXiv API**: Preprint paper search
- **CrossRef API**: DOI resolution

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Internet connection for API access
- 4GB+ RAM recommended
- 2GB+ free disk space

### Required API Keys
- **Groq API Key**: Sign up at [Groq](https://groq.com) and get your API key

### Python Dependencies
