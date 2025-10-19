import os
import time
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

import main

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI YouTube Assistant — Summary & Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- GLOBAL STYLING ---
APP_STYLE = """
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #0f172a; /* Dark text for readability */
    }
    section[data-testid="stSidebar"] {
        background: rgba(241, 245, 249, 0.95);
        backdrop-filter: blur(18px);
        border-right: 1px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stTextInput>label,
    section[data-testid="stSidebar"] .stSelectbox>label,
    section[data-testid="stSidebar"] .stSlider>label,
    section[data-testid="stSidebar"] .stRadio>label {
        color: #0f172a !important; /* Ensure all sidebar text is dark */
    }
    section[data-testid="stSidebar"] .stButton>button {
        width: 100%;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 18px 60px rgba(13, 17, 23, 0.05);
        margin-bottom: 1.5rem;
    }
    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        border: 1px solid #cbd5e1;
        background: #f1f5f9;
        font-size: 0.86rem;
    }
    .pill-button>button {
        border-radius: 999px !important;
        border: none;
        background: linear-gradient(120deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pill-button>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(59, 130, 246, 0.2);
    }
    .secondary-button>button {
        border-radius: 999px !important;
        border: 1px solid #cbd5e1 !important;
        background: transparent !important;
        color: #334155 !important;
        padding: 0.55rem 1.4rem;
    }
    .stChatMessage {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>div {
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        color: #0f172a;
    }
    .stSlider>div>div>div[data-baseweb="slider"]>div>div {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }
    .stAlert {
        border-radius: 16px;
    }
</style>
"""

st.markdown(APP_STYLE, unsafe_allow_html=True)


# --- ENVIRONMENT AND STATE ---
load_dotenv()

DEFAULT_STATE: Dict[str, Optional[object]] = {
    "video_id": None,
    "video_details": None,
    "transcript": None,
    "vector_store": None,
    "rag_chain": None,
    "summary": None,
    "messages": [],
    "response_latency": None,
    "cache_hit": False,
}


def initialize_session_state() -> None:
    for key, default in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = (
                default.copy() if isinstance(default, list) else default
            )


def reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["rag_chain"] = None
    st.session_state["summary"] = None
    st.session_state["response_latency"] = None


def extract_highlights(summary: str, limit: int = 4) -> List[str]:
    lines = [line.strip("•- ") for line in summary.splitlines() if line.strip()]
    bullets = [line for line in lines if line.startswith(("*", "-", "•"))]
    if not bullets:
        bullets = lines
    return bullets[:limit]


def format_number(value: Optional[int]) -> str:
    if not value:
        return "—"
    return f"{value:,}"


SUMMARY_MODES: Dict[str, str] = {
    "Quick": (
        "Create a concise summary under 150 words. Present the top three takeaways as bullet "
        "points followed by one-sentence context for each."
    ),
    "Balanced": (
        "Write a well-structured summary (180-240 words) divided into sections: Overview, Key "
        "Insights, and Notable Details. Use bullet lists where appropriate."
    ),
    "In-depth": (
        "Provide an in-depth analytical summary (up to 320 words). Include sections for Overview, "
        "Main Arguments, Supporting Evidence, and Recommended Actions."
    ),
}


initialize_session_state()


def process_video(youtube_url: str) -> None:
    if not youtube_url:
        st.warning("📺 Please enter a YouTube URL to begin.")
        return

    video_id = main.yt_id(youtube_url)
    if not video_id:
        st.error("❌ That doesn’t look like a valid YouTube URL.")
        return

    if (
        video_id == st.session_state.get("video_id")
        and st.session_state.get("transcript")
    ):
        st.session_state["cache_hit"] = True
        st.info("✅ Using the cached transcript for this video.")
        return

    progress = st.progress(0, text="Fetching video metadata…")

    st.session_state["cache_hit"] = False
    st.session_state["video_id"] = video_id
    reset_conversation()

    video_details = main.get_video_details(video_id)
    progress.progress(20, text="Downloading transcript…")
    transcript = main.get_transcript(video_id)

    if not transcript:
        progress.empty()
        st.error("❌ No transcript available. Try another video with subtitles enabled.")
        st.session_state["video_details"] = video_details
        st.session_state["transcript"] = None
        st.session_state["vector_store"] = None
        return

    st.session_state["video_details"] = video_details
    st.session_state["transcript"] = transcript
    st.session_state["vector_store"] = None
    progress.progress(100, text="Video processed! Ready to summarize.")
    time.sleep(0.2)
    progress.empty()
    st.success("Video processed! Choose your settings and generate a summary.")


def build_chain_and_summary(
    model: str, summary_mode: str, temperature: float = 0.7, k_value: int = 6
) -> None:
    transcript = st.session_state.get("transcript")
    if not transcript:
        st.warning("📼 Please process a video first.")
        return

    if not st.session_state.get("vector_store"):
        with st.spinner("Embedding transcript and building the knowledge base…"):
            st.session_state["vector_store"] = main.create_vector_store(transcript)

    with st.spinner("Creating the reasoning chain and generating the summary…"):
        start = time.perf_counter()
        st.session_state["rag_chain"] = main.create_rag_chain(
            st.session_state["vector_store"],
            temperature=temperature,
            k=k_value,
            model=model,
        )

        summary_prompt = SUMMARY_MODES.get(summary_mode, SUMMARY_MODES["Balanced"])
        summary = st.session_state["rag_chain"].invoke(summary_prompt)
        st.session_state["summary"] = summary
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Summary ready! Ask me anything about the video. 💬",
            }
        ]
        st.session_state["response_latency"] = time.perf_counter() - start
        st.toast("✅ Summary generated. Scroll down to explore!", icon="✅")


# --- SIDEBAR ---
with st.sidebar:
    st.title("🎬 AI YouTube Assistant")
    st.caption("Summaries, insights, and Q&A on top of any captioned video.")
    youtube_url = st.text_input(
        "Video URL", placeholder="https://www.youtube.com/watch?v=…"
    )

    st.button(
        "Process Video", on_click=process_video, args=(youtube_url,), type="primary"
    )

    if st.session_state.get("transcript"):
        st.markdown("---")
        st.subheader("⚙️ Model Controls")
        model_choice = st.selectbox(
            "Model",
            options=main.SUPPORTED_MODELS,
            index=0,
            help="Choose the language model served via Groq.",
        )
        # Fixed values for temperature and k_value (controlled by developer only)
        temperature = 0.7
        k_value = 6
        
        summary_mode = st.radio(
            "Summary style", list(SUMMARY_MODES.keys()), index=1, horizontal=False
        )

        st.button(
            "Generate / Refresh Summary",
            on_click=build_chain_and_summary,
            args=(model_choice, summary_mode),
            key="summarize_btn",
            type="primary",
        )

    st.markdown("---")
    st.info(
        "💡 **Tips**\n"
        "Ask follow-up questions such as _Who is the speaker?_ or _List the main steps_.\n"
        "Longer videos (>30 min) can take a little longer to process."
    )


# --- MAIN CONTENT ---
st.title("AI YouTube Assistant 🎥🤖")

hero_container = st.container()

if not st.session_state.get("transcript"):
    with hero_container:
        st.markdown(
            """
            <div class="glass-card">
                <h3>👋 Welcome!</h3>
                <p>
                    Paste any YouTube link with subtitles and let our retrieval-augmented assistant
                    build a searchable knowledge base. Generate polished summaries, download the
                    transcript, and ask follow-up questions just like chatting with the video.
                </p>
                <ul>
                    <li>📼 Supports public videos with transcripts</li>
                    <li>🧠 Powered by Groq-hosted LLMs and FAISS vector search</li>
                    <li>⚡ Designed for analysts, students, and creators</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


details = st.session_state.get("video_details")
if details:
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        header_cols = st.columns([1, 2], gap="large")
        with header_cols[0]:
            if details.get("thumbnail_url"):
                st.image(details["thumbnail_url"], use_container_width=True)
        with header_cols[1]:
            st.subheader(details.get("title", "Untitled Video"))
            if details.get("author"):
                st.caption(f"By {details['author']}")
            if st.session_state.get("video_id"):
                st.video(
                    f"https://www.youtube.com/watch?v={st.session_state['video_id']}"
                )

            metrics_cols = st.columns(3)
            metrics_cols[0].markdown(
                f"<div class='metric-chip'>⏱️ {details.get('duration', '—')}</div>",
                unsafe_allow_html=True,
            )
            metrics_cols[1].markdown(
                f"<div class='metric-chip'>👁️ {format_number(details.get('views'))} views</div>",
                unsafe_allow_html=True,
            )
            metrics_cols[2].markdown(
                f"<div class='metric-chip'>📅 {details.get('publish_date', '—')}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.get("summary"):
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📝 Video Summary")

        highlights = extract_highlights(st.session_state["summary"])
        if highlights:
            st.markdown("**Highlights**")
            st.markdown("\n".join(f"• {item}" for item in highlights))

        with st.expander("Read full summary", expanded=True):
            st.markdown(st.session_state["summary"])

        download_cols = st.columns(2)
        download_cols[0].download_button(
            label="Download Summary",
            data=st.session_state["summary"].encode("utf-8"),
            file_name="video_summary.txt",
            mime="text/plain",
        )
        if st.session_state.get("transcript"):
            download_cols[1].download_button(
                label="Download Transcript",
                data=st.session_state["transcript"].encode("utf-8"),
                file_name="video_transcript.txt",
                mime="text/plain",
            )

        if st.session_state.get("response_latency"):
            st.caption(
                f"Summary generated in {st.session_state['response_latency']:.2f} seconds"
                f"{' ⏱️ (cached)' if st.session_state.get('cache_hit') else ''}"
            )

        st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.get("rag_chain"):
    st.markdown("---")
    st.subheader("💬 Ask follow-up questions")

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button("Clear conversation", key="clear_chat", type="secondary"):
        reset_conversation()
        st.toast("Chat history cleared.")

    if st.session_state.get("messages"):
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the video…")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                start = time.perf_counter()
                response = st.session_state["rag_chain"].invoke(prompt)
                latency = time.perf_counter() - start
                st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.session_state["response_latency"] = latency

        st.caption(f"Latest answer generated in {latency:.2f} seconds")
