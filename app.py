import streamlit as st
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="TARK | v2.0",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Config:
    PDF_PATH = "satellite_manual.pdf"
    DB_DIR = "chroma_db_tark"
    MODEL_NAME = "phi3"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    WHISPER_MODEL = "base"
    SAMPLE_RATE = 44100
    RECORD_SECONDS = 5

# --- 2. THEME ---
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

.stApp {
    background-color: #050f1e;
    color: #e6e6e6;
    font-family: 'Rajdhani', sans-serif;
}

/* MAIN TITLE (reusing sidebar style) */
.sidebar-header {
    font-family: 'Orbitron';
    font-size: 1.4em;
    color: #FF9933;
    text-align: center;
    padding-top: 20px;
}

/* LARGE TITLE OVERRIDE */
.main-header {
    font-family: 'Orbitron';
    font-size: 5em;
    color: #FF9933;
    text-align: center;
    text-shadow: 0 0 25px rgba(255,153,51,0.4);
    margin-bottom: 10px;
}

/* SUBTITLE */
.subtitle-container {
    padding: 25px 0 30px 0;
    margin-bottom: 40px;
}
.main-subtitle {
    font-size: 1.6em;
    color: #FF9933;
    text-align: center;
    letter-spacing: 5px;
    text-transform: uppercase;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #050f1e;
    border-right: 2px solid #FF9933;
}
.mission-box {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid #FF9933;
    padding: 15px;
    margin-bottom: 20px;
}
.highlight {
    color: #FF9933;
    font-weight: bold;
}

/* CHAT */
.stChatMessage {
    background-color: #162a4d;
    border: 1px solid #4a6fa5;
    border-radius: 4px;
}

/* BUTTONS */
div.stButton > button {
    background: transparent;
    border: 1px solid #FF9933;
    color: #FF9933;
    font-family: 'Orbitron';
    font-size: 1.2em;
    width: 100%;
}
</style>""", unsafe_allow_html=True)

# --- 3. BACKEND ---
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_system():
    if not os.path.exists(Config.PDF_PATH):
        return None

    loader = PyPDFLoader(Config.PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=Config.DB_DIR)

    template = """Use the following context to answer the question.
If a specific value is requested, prioritize tables.
If unsure, say you don't know.

Context: {context}
Question: {question}

Answer:"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm = Ollama(model=Config.MODEL_NAME)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

@st.cache_resource
def load_whisper():
    return whisper.load_model(Config.WHISPER_MODEL)

def record_audio(filename="query.wav"):
    with st.spinner("🔴 RECORDING SECURE COMMS..."):
        audio = sd.rec(
            int(Config.RECORD_SECONDS * Config.SAMPLE_RATE),
            samplerate=Config.SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )
        sd.wait()
        wav.write(filename, Config.SAMPLE_RATE, audio)
    return filename

def transcribe_audio(file, model):
    return model.transcribe(file, fp16=False)["text"].strip()

# --- HELPER FUNCTIONS ---
def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def append_user_message(content):
    st.session_state.messages.append({"role": "user", "content": content})
    st.rerun()

def generate_ai_response():
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("ANALYZING FLIGHT DATA..."):
                res = st.session_state.qa_chain.invoke(
                    {"query": st.session_state.messages[-1]["content"]}
                )
                answer = res["result"]
                pages = sorted({f"p.{d.metadata.get('page',0)+1}" for d in res["source_documents"]})
                output = f"{answer}\n\n**SOURCE:** {', '.join(pages)}"
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})

# --- 4. UI ---
def main():
    with st.sidebar:
        st.markdown("---")
        st.markdown("**MISSION PROFILE:**")
        st.markdown("""
        <div class="mission-box">
        Deep space missions encounter <span class="highlight">20+ minute delays</span>.
        Real-time Earth support is impossible.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**SYSTEM CAPABILITY:**")
        st.markdown("""
        <div class="mission-box">
        Offline <span class="highlight">Neural Reasoning</span>
        using onboard flight manuals.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.success("✅ MANUAL SECURE" if os.path.exists(Config.PDF_PATH) else "❌ MANUAL MISSING")
        st.metric("LATENCY", "0.02s")

    st.markdown('<div class="main-header">PROJECT T.A.R.K.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle-container">
        <div class="main-subtitle">Technical Aerospace Reasoning Kernel</div>
    </div>
    """, unsafe_allow_html=True)

    if "qa_chain" not in st.session_state:
        with st.spinner("INITIALIZING NEURAL NETWORKS..."):
            st.session_state.qa_chain = load_system()
            st.session_state.whisper = load_whisper()

    # Display chat history and AI response
    display_chat_history()
    generate_ai_response()

    # ---- INPUT AREA (MIC + CHAT INPUT ALIGNED) ----
    col1, col2 = st.columns([1, 6])

    with col1:
        mic_clicked = st.button("🎙️ COMMS")  # store click

    with col2:
        user_text = st.chat_input("Enter emergency protocol query...")  # chat input

    # Handle mic input
    if mic_clicked:
        audio = record_audio()
        text = transcribe_audio(audio, st.session_state.whisper)
        if text:
            append_user_message(text)

    # Handle chat input
    if user_text:
        append_user_message(user_text)

if __name__ == "__main__":
    main()
