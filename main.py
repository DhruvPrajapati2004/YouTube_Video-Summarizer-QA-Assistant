import html
import json
import os
import re
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

try:
    from pytube import YouTube
except Exception:   
    YouTube = None
try:
    import yt_dlp
except Exception:
    yt_dlp = None

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

PREFERRED_LANGUAGES = [
    "en",
    "en-US",
    "en-GB",
    "en-IN",
    "en-AU",
    "en-CA",
    "en-IE",
    "en-NZ",
    "en-UK",
    "en-ZA",
    "en-KE",
    "en-SG",
    "en-419",
]


SUPPORTED_MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
]

# Mapping from alias → canonical model name (must be a name in SUPPORTED_MODELS)
MODEL_ALIASES = {
    # Llama‑3.1 8B
    "llama3-8b-8192": "llama-3.1-8b-instant",
    # Llama‑3.1 70B
    "llama3-70b-8192": "llama-3.1-70b-versatile",
    # Mixtral‑8x7b is an alternate name people sometimes use for GPT‑OSS‑20B
    "mixtral-8x7b-32768": "openai/gpt-oss-20b",
    # Common short forms
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gemma-9b": "gemma2-9b-it",
}

# Default model to use if no explicit choice is made
DEFAULT_MODEL = SUPPORTED_MODELS[0]

def _clean_transcript_chunks(chunks: list[dict]) -> str | None:
    cleaned_chunks = [
        chunk.get("text", "").replace("\n", " ").strip()
        for chunk in chunks
        if chunk.get("text", "").strip()
    ]
    transcript = " ".join(cleaned_chunks).strip()
    return transcript or None


def _parse_json3_payload(payload: str) -> str | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    segments: list[str] = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            text = seg.get("utf8", "").replace("\n", " ").strip()
            if text and text != "\n":
                segments.append(html.unescape(text))
    joined = " ".join(segments).strip()
    return joined or None


def _parse_vtt_payload(payload: str) -> str | None:
    segments: list[str] = []
    for line in payload.splitlines():
        content = line.strip()
        if not content or content.startswith("WEBVTT"):
            continue
        if "-->" in content or content.isdigit():
            continue
        content = re.sub(r"<[^>]+>", "", content)
        content = html.unescape(content).strip()
        if content:
            segments.append(content)
    joined = " ".join(segments).strip()
    return joined or None


def _download_caption_text(url: str) -> str | None:
    try:
        with urlopen(url) as response:  # nosec: trusted source (YouTube)
            payload_bytes = response.read()
    except Exception as exc:
        print(f"Failed to download caption track: {exc}")
        return None

    payload = payload_bytes.decode("utf-8", errors="ignore")
    if payload.strip().startswith("{"):
        return _parse_json3_payload(payload)
    return _parse_vtt_payload(payload)


def _transcript_via_yt_dlp(video_id: str) -> str | None:
    if yt_dlp is None:
        return None

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",
        "subtitleslangs": PREFERRED_LANGUAGES,
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        print(f"yt-dlp transcript lookup failed: {exc}")
        return None

    sources = []
    for key in ("requested_subtitles", "subtitles", "automatic_captions"):
        data = info.get(key) or {}
        if isinstance(data, dict):
            sources.append(data)

    for language in PREFERRED_LANGUAGES:
        for source in sources:
            tracks = source.get(language)
            if not tracks:
                continue
            if isinstance(tracks, dict):
                tracks = [tracks]
            for track in tracks:
                track_url = track.get("url")
                if not track_url:
                    continue
                text = _download_caption_text(track_url)
                if text:
                    return text

    for source in sources:
        for tracks in source.values():
            if isinstance(tracks, dict):
                tracks = [tracks]
            for track in tracks:
                track_url = track.get("url")
                if not track_url:
                    continue
                text = _download_caption_text(track_url)
                if text:
                    return text

    return None


def yt_id(url: str) -> str | None:
    """Extract the video id from a YouTube URL."""
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        return query.get("v", [None])[0]
    return None


def get_transcript(video_id: str) -> str | None:
    """Return an English transcript for the given video or ``None`` if unavailable."""

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    except TranscriptsDisabled:
        print("No captions available for this video (explicitly disabled).")
        return _transcript_via_yt_dlp(video_id)
    except VideoUnavailable:
        print("The video is unavailable (private, deleted, or region-blocked).")
        return None
    except CouldNotRetrieveTranscript as exc:
        print(f"YouTubeTranscriptApi could not retrieve transcript: {exc}")
        return _transcript_via_yt_dlp(video_id)
    except Exception as exc:
        print(f"Unexpected transcript retrieval error: {exc}")
        return _transcript_via_yt_dlp(video_id)

    selected_transcript = None

    for finder_name in (
        "find_transcript",
        "find_manually_created_transcript",
        "find_generated_transcript",
    ):
        finder = getattr(transcripts, finder_name, None)
        if finder is None:
            continue
        try:
            selected_transcript = finder(PREFERRED_LANGUAGES)
            if selected_transcript:
                break
        except NoTranscriptFound:
            continue

    if not selected_transcript:
        for transcript in transcripts:
            code = (transcript.language_code or "").lower()
            if code.startswith("en") or code.endswith("en") or "en" in code.split("-"):
                selected_transcript = transcript
                break

    if not selected_transcript:
        for transcript in transcripts:
            if not transcript.is_translatable:
                continue
            try:
                selected_transcript = transcript.translate("en")
                if selected_transcript:
                    break
            except Exception:
                continue

    fetched_chunks = None
    if selected_transcript:
        try:
            fetched_chunks = selected_transcript.fetch()
        except Exception as exc:
            print(f"Failed to fetch transcript chunks: {exc}")
            fetched_chunks = None

    if fetched_chunks:
        transcript_text = _clean_transcript_chunks(fetched_chunks)
        if transcript_text:
            return transcript_text

    print("Falling back to yt-dlp transcript extraction.")
    transcript_text = _transcript_via_yt_dlp(video_id)
    if transcript_text:
        return transcript_text

    try:
        fetched_chunks = YouTubeTranscriptApi.get_transcript(
            video_id, languages=PREFERRED_LANGUAGES
        )
        transcript_text = _clean_transcript_chunks(fetched_chunks)
        if transcript_text:
            return transcript_text
    except Exception as exc:
        print(f"Direct transcript download failed: {exc}")

    print("No transcripts available for this video after all fallbacks.")
    return None


def _format_duration(seconds: int | None) -> str | None:
    if not seconds:
        return None
    try:
        return str(timedelta(seconds=int(seconds)))
    except Exception:
        return None


def _format_upload_date(upload_date: str | None) -> str | None:
    if not upload_date:
        return None
    try:
        if isinstance(upload_date, str) and len(upload_date) == 8:
            return datetime.strptime(upload_date, "%Y%m%d").strftime("%d %b %Y")
        if hasattr(upload_date, "strftime"):
            return upload_date.strftime("%d %b %Y")
    except Exception:
        return None
    return None


def _details_via_yt_dlp(video_id: str) -> dict | None:
    if yt_dlp is None:
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        print(f"yt-dlp could not fetch metadata: {exc}")
        return None

    return {
        "title": info.get("title"),
        "thumbnail_url": info.get("thumbnail"),
        "author": info.get("uploader"),
        "channel_url": info.get("channel_url"),
        "duration": _format_duration(info.get("duration")),
        "publish_date": _format_upload_date(info.get("upload_date")),
        "views": info.get("view_count"),
    }


def _details_via_oembed(video_id: str) -> dict | None:
    url = (
        "https://www.youtube.com/oembed?format=json&url="
        f"https://www.youtube.com/watch?v={video_id}"
    )
    try:
        with urlopen(url) as response:  # nosec: URL is controlled and uses https
            payload = json.load(response)
    except Exception as exc:
        print(f"YouTube oEmbed lookup failed: {exc}")
        return None

    return {
        "title": payload.get("title"),
        "thumbnail_url": payload.get("thumbnail_url"),
        "author": payload.get("author_name"),
        "channel_url": payload.get("author_url"),
        "duration": None,
        "publish_date": None,
        "views": None,
    }


def get_video_details(video_id: str) -> dict | None:
    """Fetch video metadata used for the UI cards."""

    details = _details_via_yt_dlp(video_id)
    if details:
        return details

    details = _details_via_oembed(video_id)
    if details:
        return details

    if YouTube is None:
        print("pytube is unavailable and yt_dlp failed to fetch metadata.")
        return None

    try:
        yt = YouTube(
            f"https://www.youtube.com/watch?v={video_id}",
            use_oauth=False,
            allow_oauth_cache=True,
        )
    except Exception as exc:
        print(f"Could not initialize YouTube object: {exc}")
        return None

    try:
        details = {
            "title": yt.title,
            "thumbnail_url": yt.thumbnail_url,
            "author": yt.author,
            "channel_url": getattr(yt, "channel_url", None),
            "duration": _format_duration(getattr(yt, "length", None)),
            "publish_date": _format_upload_date(getattr(yt, "publish_date", None)),
            "views": getattr(yt, "views", None),
        }
    except Exception as exc:
        print(f"pytube metadata lookup failed: {exc}")
        return None

    return details


def create_vector_store(transcript: str):
    """Create a FAISS vector store from the transcript."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def create_rag_chain(
    vector_store,
    temperature: float = 0.7,
    k: int = 6,
    model: str = DEFAULT_MODEL,
):
    """Create the full RAG chain with configurable model settings."""
    k = max(1, int(k))
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    resolved_model = MODEL_ALIASES.get(model, model)
    if resolved_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported Groq model '{model}'. Choose one of: {', '.join(SUPPORTED_MODELS)}"
        )

    llm = ChatGroq(
        model=resolved_model,
        temperature=float(temperature),
        groq_api_key=api_key,
    )

    def combine_docs(retrieved_docs):
        return " ".join(doc.page_content for doc in retrieved_docs)

    prompt_template = """
    You are an intelligent and friendly AI assistant specialized in summarizing and answering questions about YouTube videos.

    🎯 Your goal:
    Be helpful and engaging when answering questions about the YouTube video.

    🤖 Types of Questions:
    1. For greetings like "hi", "hello", or general questions about the assistant:
       - Respond in a friendly manner.
       - You can briefly mention what the video is about based on the context.
       - Example: "Hello! I'm your AI assistant for this video about [topic]. How can I help you understand it better?"

    2. For video content questions:
       - Provide accurate answers using only information from the transcript.
       - If not found in the transcript, say "I don't see that mentioned in the video" and try to suggest related topics that were covered.

    ⚠️ Important Guidelines:
    - For video-specific questions, only use information from the provided transcript context.
    - Be conversational for greetings and simple questions.
    - Avoid making up information not present in the transcript.

    🧠 Response Guidelines:
    1. Be **clear, direct, and descriptive** — explain briefly *why* if it adds clarity.
    2. If multiple points are relevant, use **• bullet points** or **numbered lists**.
    3. Highlight key ideas or phrases using **bold** for readability.
    4. Keep tone friendly and informative — feel free to add light emojis like 🙂
    (Examples: 🤖, 💡, 🎯, 📚, 🌟, 🚀, 🧠, 🔥).
    5. Avoid filler intros (e.g., "According to the transcript" or "Yes, the transcript says...").
    6. Do not restate the question unless needed for clarity.

    ---

    📜 Transcript Context:
    {context}

    ❓ Question:
    {question}

    💬 Answer:
"""
    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(combine_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

