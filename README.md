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

## 🚀 Features

- 🔍 Search papers from ArXiv and Semantic Scholar
- 🧠 Summarize with Groq API (LLM)
- 🎧 Generate MP3 summaries using TTS (`gTTS` or `pyttsx3`)
- 📁 Organize papers by topic
- 📝 Auto-generate APA citations

---

## 🖥️ Requirements

- Python 3.10 or 3.11
- pip
- virtualenv (recommended)

### 🛠️ System dependencies (Linux)

bash
sudo apt update
sudo apt install espeak ffmpeg libespeak1

⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/matrixkel/research_paper_audio_bot.git
cd research_paper_audio_bot
2. Create & Activate Virtual Environment
bash
Copy
Edit
python3.11 -m venv venv
source venv/bin/activate
If you're using Python 3.10, change python3.11 to python3.10.



4. Install Python Dependencies
bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
🔐 Environment Variables
Create a .env file in the root directory with your Groq API key:

ini
Copy
Edit
# .env
GROQ_API_KEY=your_groq_api_key_here
You can get a key from: https://console.groq.com/keys

▶️ Run the App
bash
Copy
Edit
streamlit run app.py --server.port 5000
Then open your browser at:

arduino
Copy
Edit
http://localhost:5000
🔊 Text-to-Speech (TTS)
This app supports two TTS engines:

Engine	Type	Requirements
gTTS	Online	Internet connection
pyttsx3	Offline	espeak, ffmpeg

Both are installed by default. The app prefers gTTS.

📦 Requirements.txt
txt
Copy
Edit
streamlit
openai
requests
python-dotenv
aiohttp
PyMuPDF
trafilatura
justext
lxml_html_clean
pandas
groq
pyttsx3
gTTS
🗂️ Project Structure
bash
Copy
Edit
├── app.py
├── .env
├── requirements.txt
├── README.md
├── storage/
│   └── audio/                 # Saved MP3 files
├── agents/
│   ├── base.py
│   ├── coordinator.py
│   ├── paper_discovery.py
│   ├── paper_processing.py
│   ├── summarization.py
│   ├── synthesis.py
│   └── text_to_speech.py
├── utils/
│   └── data_models.py

### Required API Keys
- **Groq API Key**: Sign up at [Groq](https://groq.com) and get your API key

