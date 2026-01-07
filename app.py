import streamlit as st
import os
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="TARK | Offline AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Config:
    PDF_PATH = "satellite_manual.pdf"
    DB_DIR = "chroma_db_data"
    MODEL_NAME = "phi3"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    WHISPER_MODEL = "base"
    SAMPLE_RATE = 44100
    RECORD_SECONDS = 5

# --- 2. PROFESSIONAL CSS ---
st.markdown("""
<style>
    /* MAIN THEME: Deep Space Navy & Professional Orange */
    
    /* Hide Default Streamlit UI */
    [data-testid="stDecoration"] {display: none;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B192E; /* Deep Space Navy */
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: #E0E6ED !important; /* Steel Blue Text */
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #FF9F1C !important; /* Tech Orange */
    }
    [data-testid="stMetricLabel"] {
        color: #8D99AE !important;
    }
    
    /* Chat Bubbles */
    /* Assistant: Deep Navy */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #0B192E !important;
    }
    
    /* User: Tech Orange */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #FF9F1C !important;
    }
    
    /* Story Box */
    .story-box {
        background-color: rgba(255, 159, 28, 0.1);
        border-left: 4px solid #FF9F1C;
        padding: 15px;
        border-radius: 4px;
        font-size: 13px;
        color: #E0E6ED;
        margin-top: 20px;
        line-height: 1.5;
    }
    
    /* Main Headers */
    h1 {
        color: #0B192E;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Input Box */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND (Cached) ---
@st.cache_resource
def load_system():
    if not os.path.exists(Config.PDF_PATH):
        return None
    
    loader = PyPDFLoader(Config.PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=Config.DB_DIR)
    llm = Ollama(model=Config.MODEL_NAME)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True
    )

@st.cache_resource
def load_whisper():
    return whisper.load_model(Config.WHISPER_MODEL)

def record_audio(filename="query.wav"):
    with st.spinner("Recording Audio Link..."):
        audio_data = sd.rec(int(Config.RECORD_SECONDS * Config.SAMPLE_RATE), 
                            samplerate=Config.SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wav.write(filename, Config.SAMPLE_RATE, audio_data)
    return filename

def transcribe_audio(filename, model):
    result = model.transcribe(filename, fp16=False)
    return result["text"].strip()

# --- 4. MAIN UI ---
def main():
    # --- SIDEBAR (Context & Diagnostics) ---
    with st.sidebar:
        st.markdown("## Mission Control")
        st.markdown("IDSS Interface Module")
        st.markdown("---")
        
        # Diagnostics
        st.markdown("### System Diagnostics")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Link", "ACTIVE")
        with c2:
            st.metric("Latency", "0ms")
            
        st.caption(f"**Loaded Manual:** {Config.PDF_PATH}")
        
        # Origin Story
        st.markdown("### Project Origin")
        st.markdown("""
        <div class="story-box">
        <b>The Challenge:</b><br>
        Deep space missions (Mars) face 20-minute signal delays. Astronauts cannot rely on Earth for instant technical support.<br><br>
        <b>The Solution:</b><br>
        TARK is an autonomous, offline reasoning kernel. It ingests flight manuals locally, providing instant voice-activated answers during critical operations without internet access.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("Wipe Session Memory", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN PAGE (The Software) ---
    # UPDATED: "TARK" is now the big header
    st.title("TARK")
    st.caption("Technical Aerospace Reasoning Kernel • Secure Local Environment")
    
    # State Init
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "TARK Online. Flight manuals ingested. Ready for query."}]

    # Load Backend
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = load_system()
        st.session_state.whisper_model = load_whisper()

    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Area
    st.markdown("---")
    col1, col2 = st.columns([1, 10])
    
    with col1:
        if st.button("🎙️", help="Voice Command", use_container_width=True):
            audio_file = record_audio()
            text = transcribe_audio(audio_file, st.session_state.whisper_model)
            if text:
                st.session_state.messages.append({"role": "user", "content": text})
                st.rerun()

    with col2:
        if prompt := st.chat_input("Input flight parameter query..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    # AI Response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing Technical Data..."):
                try:
                    res = st.session_state.qa_chain.invoke({"query": st.session_state.messages[-1]["content"]})
                    answer = res['result']
                    sources = sorted(list(set([f"p.{doc.metadata.get('page',-1)+1}" for doc in res['source_documents']])))
                    
                    full_response = f"{answer}\n\n**Ref:** {', '.join(sources)}"
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception:
                    st.error("Retrieval failed.")

if __name__ == "__main__":
    main()