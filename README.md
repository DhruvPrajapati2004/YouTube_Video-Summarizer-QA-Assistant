# YouTube_Video-Summarizer-QA-Assistant


**YouTube_Video-Summarizer-QA-Assistant** is an AI-powered tool that automatically extracts YouTube video transcripts, splits them into semantic chunks, generates concise summaries, and answers user queries contextually. Built using **Python, LangChain, FAISS, and Groq API**, it provides a fast, interactive pipeline for efficient video understanding and knowledge retrieval.

---

## Features

- **Transcript Extraction**: Automatically fetches YouTube captions using the `YouTubeTranscriptApi`.
- **Semantic Chunking**: Splits long transcripts into manageable chunks for efficient processing.
- **Contextual Q&A**: Answers user questions based on the video transcript using a language model via **Groq API**.
- **Summarization**: Generates concise summaries for long YouTube videos.
- **Interactive Pipeline**: Real-time retrieval and summarization to improve video understanding and learning efficiency.

---

## Tech Stack

- **Python** – Core programming language
- **LangChain** – For building LLM-based pipelines and chains
- **FAISS** – Efficient vector store for semantic search
- **Groq API** – High-speed LLM inference
- **YouTubeTranscriptApi** – Extracting video captions
- **dotenv** – Loading environment variables
- **RunnableParallel, PromptTemplate** – For chaining retrieval, summarization, and Q&A

---
