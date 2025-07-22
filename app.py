import streamlit as st
import asyncio
import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Import agents
from agents.coordinator import CoordinatorAgent
from utils.config import Config
from utils.data_models import Paper, ProcessingResult, AudioResult
from utils.helpers import ensure_directories

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Paper Analysis System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'audio_results' not in st.session_state:
        st.session_state.audio_results = {}
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = CoordinatorAgent()

def main():
    """Main application function"""
    initialize_session_state()
    ensure_directories()
    
    st.title("📚 Multi-Agent Research Paper Analysis System")
    st.markdown("Discover, analyze, and synthesize research papers with AI-powered agents")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key configuration
        groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        if groq_api_key:
            Config.GROQ_API_KEY = groq_api_key
        
        st.divider()
        
        # Topic configuration
        st.subheader("🏷️ Research Topics")
        topics_text = st.text_area(
            "Enter research topics (one per line):",
            value="\n".join(st.session_state.topics),
            height=150,
            help="Define topics for paper classification and synthesis"
        )
        
        if st.button("Update Topics"):
            st.session_state.topics = [topic.strip() for topic in topics_text.split('\n') if topic.strip()]
            st.success(f"Updated {len(st.session_state.topics)} topics")
        
        if st.session_state.topics:
            st.write("**Current Topics:**")
            for i, topic in enumerate(st.session_state.topics, 1):
                st.write(f"{i}. {topic}")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Paper Input", "📊 Analysis Results", "🎧 Audio Summaries", "📖 Citations"])
    
    with tab1:
        handle_paper_input()
    
    with tab2:
        display_analysis_results()
    
    with tab3:
        display_audio_summaries()
    
    with tab4:
        display_citations()

def handle_paper_input():
    """Handle various paper input methods"""
    st.header("Paper Input Methods")
    
    input_method = st.selectbox(
        "Choose input method:",
        ["Search Papers", "Upload PDF", "Enter DOI", "Enter URL"]
    )
    
    if input_method == "Search Papers":
        handle_paper_search()
    elif input_method == "Upload PDF":
        handle_pdf_upload()
    elif input_method == "Enter DOI":
        handle_doi_input()
    elif input_method == "Enter URL":
        handle_url_input()

def handle_paper_search():
    """Handle paper search functionality"""
    st.subheader("🔍 Search Academic Papers")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Search query:", placeholder="Enter keywords, topics, or research areas")
    
    with col2:
        max_papers = st.number_input("Max papers:", min_value=1, max_value=50, value=10)
    
    # Search filters
    with st.expander("🔧 Search Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year_from = st.number_input("From year:", min_value=1900, max_value=2025, value=2020)
        
        with col2:
            year_to = st.number_input("To year:", min_value=1900, max_value=2025, value=2025)
        
        with col3:
            source = st.selectbox("Source:", ["Both", "Semantic Scholar", "ArXiv"])
    
    if st.button("🔍 Search Papers", type="primary"):
        if query and st.session_state.topics:
            with st.spinner("Searching for papers..."):
                try:
                    search_params = {
                        'query': query,
                        'max_papers': max_papers,
                        'year_from': year_from,
                        'year_to': year_to,
                        'source': source,
                        'topics': st.session_state.topics
                    }
                    
                    # Use coordinator to handle the search and processing
                    results = asyncio.run(st.session_state.coordinator.process_search_request(search_params))
                    
                    if results['success']:
                        st.session_state.papers.extend(results['papers'])
                        st.session_state.processing_results.update(results['processing_results'])
                        st.success(f"Found and processed {len(results['papers'])} papers!")
                        st.rerun()
                    else:
                        st.error(f"Search failed: {results['error']}")
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
        else:
            if not query:
                st.error("Please enter a search query")
            if not st.session_state.topics:
                st.error("Please configure research topics in the sidebar")

def handle_pdf_upload():
    """Handle PDF file upload"""
    st.subheader("📄 Upload PDF Papers")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload academic papers in PDF format"
    )
    
    if uploaded_files and st.button("📤 Process Uploaded PDFs", type="primary"):
        if st.session_state.topics:
            with st.spinner(f"Processing {len(uploaded_files)} PDF files..."):
                try:
                    # Save uploaded files
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("storage/papers", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Process with coordinator
                    results = asyncio.run(st.session_state.coordinator.process_pdf_uploads(file_paths, st.session_state.topics))
                    
                    if results['success']:
                        st.session_state.papers.extend(results['papers'])
                        st.session_state.processing_results.update(results['processing_results'])
                        st.success(f"Successfully processed {len(results['papers'])} PDF files!")
                        st.rerun()
                    else:
                        st.error(f"Processing failed: {results['error']}")
                        
                except Exception as e:
                    st.error(f"Upload processing error: {str(e)}")
        else:
            st.error("Please configure research topics in the sidebar first")

def handle_doi_input():
    """Handle DOI input"""
    st.subheader("🔗 Enter DOI")
    
    doi_input = st.text_area(
        "Enter DOI(s) (one per line):",
        placeholder="10.1038/nature12373\n10.1126/science.1234567",
        height=100
    )
    
    if doi_input and st.button("📥 Process DOIs", type="primary"):
        if st.session_state.topics:
            dois = [doi.strip() for doi in doi_input.split('\n') if doi.strip()]
            
            with st.spinner(f"Processing {len(dois)} DOI(s)..."):
                try:
                    results = asyncio.run(st.session_state.coordinator.process_dois(dois, st.session_state.topics))
                    
                    if results['success']:
                        st.session_state.papers.extend(results['papers'])
                        st.session_state.processing_results.update(results['processing_results'])
                        st.success(f"Successfully processed {len(results['papers'])} papers from DOIs!")
                        st.rerun()
                    else:
                        st.error(f"DOI processing failed: {results['error']}")
                        
                except Exception as e:
                    st.error(f"DOI processing error: {str(e)}")
        else:
            st.error("Please configure research topics in the sidebar first")

def handle_url_input():
    """Handle URL input"""
    st.subheader("🌐 Enter URLs")
    
    url_input = st.text_area(
        "Enter URL(s) (one per line):",
        placeholder="https://arxiv.org/abs/1234.5678\nhttps://www.nature.com/articles/nature12373",
        height=100
    )
    
    if url_input and st.button("🌍 Process URLs", type="primary"):
        if st.session_state.topics:
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            with st.spinner(f"Processing {len(urls)} URL(s)..."):
                try:
                    results = asyncio.run(st.session_state.coordinator.process_urls(urls, st.session_state.topics))
                    
                    if results['success']:
                        st.session_state.papers.extend(results['papers'])
                        st.session_state.processing_results.update(results['processing_results'])
                        st.success(f"Successfully processed {len(results['papers'])} papers from URLs!")
                        st.rerun()
                    else:
                        st.error(f"URL processing failed: {results['error']}")
                        
                except Exception as e:
                    st.error(f"URL processing error: {str(e)}")
        else:
            st.error("Please configure research topics in the sidebar first")

def display_analysis_results():
    """Display paper analysis results"""
    st.header("📊 Analysis Results")
    
    if not st.session_state.papers:
        st.info("No papers processed yet. Please add papers using the Paper Input tab.")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", len(st.session_state.papers))
    
    with col2:
        processed_count = len(st.session_state.processing_results)
        st.metric("Processed Papers", processed_count)
    
    with col3:
        topics_with_papers = len(set(result.topic for result in st.session_state.processing_results.values() if result.topic))
        st.metric("Topics Covered", topics_with_papers)
    
    with col4:
        synthesis_count = sum(1 for result in st.session_state.processing_results.values() if result.synthesis)
        st.metric("Syntheses Generated", synthesis_count)
    
    st.divider()
    
    # Generate syntheses button
    if st.button("🔄 Generate Topic Syntheses", type="primary"):
        with st.spinner("Generating cross-paper syntheses..."):
            try:
                synthesis_results = asyncio.run(st.session_state.coordinator.generate_syntheses(
                    st.session_state.processing_results, st.session_state.topics
                ))
                
                # Update processing results with syntheses
                for paper_id, synthesis in synthesis_results.items():
                    if paper_id in st.session_state.processing_results:
                        st.session_state.processing_results[paper_id].synthesis = synthesis
                
                st.success("Topic syntheses generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Synthesis generation error: {str(e)}")
    
    # Display results by topic
    for topic in st.session_state.topics:
        topic_papers = [
            (paper_id, result) for paper_id, result in st.session_state.processing_results.items()
            if result.topic == topic
        ]
        
        if topic_papers:
            with st.expander(f"📂 {topic} ({len(topic_papers)} papers)", expanded=True):
                
                # Display synthesis if available
                synthesis_available = any(result.synthesis for _, result in topic_papers)
                if synthesis_available:
                    st.subheader("🔗 Cross-Paper Synthesis")
                    synthesis_text = next(result.synthesis for _, result in topic_papers if result.synthesis)
                    st.markdown(synthesis_text)
                    st.divider()
                
                # Display individual papers
                st.subheader("📄 Individual Papers")
                for paper_id, result in topic_papers:
                    paper = next(p for p in st.session_state.papers if p.id == paper_id)
                    
                    with st.container():
                        st.markdown(f"**{paper.title}**")
                        st.markdown(f"*Authors: {', '.join(paper.authors)}*")
                        st.markdown(f"*Year: {paper.year} | Source: {paper.source}*")
                        
                        if result.summary:
                            with st.expander("📝 Summary", expanded=False):
                                st.markdown(result.summary)
                        
                        st.markdown("---")

def display_audio_summaries():
    """Display and generate audio summaries"""
    st.header("🎧 Audio Summaries")
    
    if not st.session_state.processing_results:
        st.info("No analysis results available. Please process papers first.")
        return
    
    # Generate audio summaries button
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🎵 Generate All Audio Summaries", type="primary"):
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    valid_papers = [
                        paper_id for paper_id, result in st.session_state.processing_results.items() 
                        if hasattr(result, 'summary') and result.summary
                    ]
                    
                    if not valid_papers:
                        st.warning("No papers with summaries available for audio generation.")
                        return
                    
                    status_text.text(f"Starting audio generation for {len(valid_papers)} papers...")
                    
                    # Generate audio with progress updates
                    audio_results = {}
                    for i, paper_id in enumerate(valid_papers):
                        progress = (i + 1) / len(valid_papers)
                        progress_bar.progress(progress)
                        status_text.text(f"Generating audio {i+1}/{len(valid_papers)}...")
                        
                        result = st.session_state.processing_results[paper_id]
                        try:
                            audio_result = asyncio.run(st.session_state.coordinator.generate_single_audio(
                                paper_id, result.summary, result.topic
                            ))
                            if audio_result:
                                audio_results[paper_id] = audio_result
                        except Exception as e:
                            st.warning(f"Failed to generate audio for paper {paper_id}: {str(e)}")
                    
                    # Clear progress indicators
                    progress_container.empty()
                    
                    if audio_results:
                        st.session_state.audio_results.update(audio_results)
                        st.success(f"Generated {len(audio_results)} audio summaries!")
                        st.rerun()
                    else:
                        st.error("No audio summaries were generated successfully.")
                    
                except Exception as e:
                    progress_container.empty()
                    st.error(f"Audio generation error: {str(e)}")
                    
    with col2:
        if st.button("🧹 Clear Audio Cache"):
            st.session_state.audio_results = {}
            st.success("Audio cache cleared!")
            st.rerun()
    
    st.divider()
    
    # Display available audio summaries
    if st.session_state.audio_results:
        st.subheader("Available Audio Summaries")
        
        for paper_id, audio_result in st.session_state.audio_results.items():
            if paper_id in st.session_state.processing_results:
                paper = next(p for p in st.session_state.papers if p.id == paper_id)
                result = st.session_state.processing_results[paper_id]
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{paper.title}**")
                        st.markdown(f"*Topic: {result.topic}*")
                    
                    with col2:
                        if hasattr(audio_result, 'file_path') and os.path.exists(audio_result.file_path):
                            try:
                                with open(audio_result.file_path, 'rb') as audio_file:
                                    audio_data = audio_file.read()
                                    st.download_button(
                                        "⬇️ Download",
                                        audio_data,
                                        file_name=f"{paper.title[:50].replace('/', '_')}.mp3",
                                        mime="audio/mp3",
                                        key=f"download_{paper_id}"
                                    )
                            except Exception as e:
                                st.error(f"Error loading audio file: {str(e)}")
                        else:
                            st.warning("Audio file not available")
                    
                    # Play audio if supported
                    if hasattr(audio_result, 'file_path') and os.path.exists(audio_result.file_path):
                        try:
                            st.audio(audio_result.file_path)
                        except Exception as e:
                            st.error(f"Error playing audio: {str(e)}")
                    
                    st.markdown("---")
    else:
        st.info("No audio summaries generated yet. Click 'Generate All Audio Summaries' to create them.")

def display_citations():
    """Display citation information"""
    st.header("📖 Citations & References")
    
    if not st.session_state.papers:
        st.info("No papers available for citation.")
        return
    
    # Citation format selector
    citation_format = st.selectbox(
        "Citation Format:",
        ["APA", "MLA", "Chicago", "BibTeX"]
    )
    
    st.subheader(f"Citations in {citation_format} format:")
    
    # Generate citations using citation manager
    try:
        citations = asyncio.run(st.session_state.coordinator.generate_citations(
            st.session_state.papers, citation_format
        ))
        
        citation_text = ""
        for i, (paper, citation) in enumerate(zip(st.session_state.papers, citations), 1):
            citation_text += f"{i}. {citation}\n\n"
        
        st.text_area(
            "Citations:",
            value=citation_text,
            height=400,
            help="Copy these citations for your research"
        )
        
        # Download citations
        st.download_button(
            "⬇️ Download Citations",
            citation_text,
            file_name=f"citations_{citation_format.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"Citation generation error: {str(e)}")
    
    st.divider()
    
    # Paper details table
    if st.session_state.papers:
        st.subheader("Paper Details")
        
        papers_data = []
        for paper in st.session_state.papers:
            processing_result = st.session_state.processing_results.get(paper.id)
            papers_data.append({
                "Title": paper.title,
                "Authors": ", ".join(paper.authors),
                "Year": paper.year,
                "Source": paper.source,
                "Topic": processing_result.topic if processing_result else "Not classified",
                "DOI": paper.doi or "N/A",
                "URL": paper.url or "N/A"
            })
        
        df = pd.DataFrame(papers_data)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
