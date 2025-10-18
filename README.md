# AI YouTube Assistant 🎥🤖

A powerful Streamlit application that uses Retrieval-Augmented Generation (RAG) and Large Language Models to summarize and answer questions about YouTube videos.

## Features

- **Video Processing**: Extracts transcripts from YouTube videos with subtitles
- **Smart Summarization**: Generate concise summaries in three different styles
  - Quick: Brief overview with top takeaways
  - Balanced: Structured summary with key insights
  - In-depth: Detailed analysis with supporting evidence
- **Interactive Q&A**: Chat with the video content using natural language
- **Responsive UI**: Modern, clean interface with elegant styling
- **Download Options**: Export summaries and transcripts as text files

## Technology Stack

- **Frontend**: Streamlit
- **Language Models**: Groq-hosted LLMs (llama-3.1, Mixtral, Gemma2)
- **Vector Storage**: FAISS for efficient semantic search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Transcript Extraction**: Multiple fallback methods with youtube_transcript_api, yt_dlp, and pytube

## Setup

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd youtube_summarizer
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv myenv
   .\myenv\Scripts\activate

   # Linux/Mac
   python -m venv myenv
   source myenv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API key:
   ```
   GROQ_API_KEY=your-groq-api-key
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Paste a YouTube URL in the input field in the sidebar
2. Click "Process Video" to extract the transcript and metadata
3. Select your preferred LLM model and summary style
4. Generate a summary with "Generate / Refresh Summary"
5. Ask follow-up questions about the video content using the chat interface

## Requirements

The app requires the following packages (specified in `requirements.txt`):
- youtube-transcript-api
- langchain-community
- langchain-openai
- faiss-cpu
- tiktoken
- python-dotenv
- deep-translator
- langchain-groq
- langchain-huggingface
- sentence-transformers
- pytube
- pyperclip
- yt_dlp (recommended for better transcript extraction)

## Acknowledgements

- LangChain for the RAG implementation
- Streamlit for the web interface
- Groq for providing LLM API access