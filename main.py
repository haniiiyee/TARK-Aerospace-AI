import os
import sys
import warnings
import time

# --- AUDIO & MATH LIBRARIES ---
# These allow the AI to "hear"
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper

# --- AI & RAG LIBRARIES ---
# These are the "Brain"
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Suppress messy warnings
warnings.filterwarnings("ignore")

# --- VISUAL STYLING (Colors make it look pro) ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- CONFIGURATION ---
class Config:
    PDF_PATH = "satellite_manual.pdf"  # This must match your file exactly
    DB_DIR = "chroma_db_data"
    MODEL_NAME = "phi3"                # The AI model we pulled earlier
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    WHISPER_MODEL = "base"             # Good balance of speed/accuracy
    SAMPLE_RATE = 44100
    RECORD_SECONDS = 6                 # How long it listens

def record_audio(filename="query.wav"):
    """Records audio from the microphone."""
    print(f"\n{Colors.RED}🎤 Recording for {Config.RECORD_SECONDS} seconds... SPEAK NOW!{Colors.ENDC}")
    
    # This records the audio
    audio_data = sd.rec(int(Config.RECORD_SECONDS * Config.SAMPLE_RATE), 
                        samplerate=Config.SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save to file
    wav.write(filename, Config.SAMPLE_RATE, audio_data)
    print(f"{Colors.GREEN}✅ Captured.{Colors.ENDC}")
    return filename

def transcribe_audio(filename):
    """Converts the recorded audio file to text using Whisper."""
    print(f"{Colors.YELLOW}📝 Transcribing...{Colors.ENDC}")
    model = whisper.load_model(Config.WHISPER_MODEL)
    result = model.transcribe(filename, fp16=False)
    return result["text"].strip()

def initialize_system():
    """Loads the PDF and prepares the AI brain."""
    print(f"\n{Colors.HEADER}--- SYSTEM STARTUP ---{Colors.ENDC}")
    
    # Check if PDF exists
    if not os.path.exists(Config.PDF_PATH):
        sys.exit(f"{Colors.RED}❌ Error: File '{Config.PDF_PATH}' not found. Please rename your PDF.{Colors.ENDC}")

    print(f"{Colors.BLUE}⚙️  Ingesting '{Config.PDF_PATH}'... (This takes a moment){Colors.ENDC}")
    
    # Load PDF
    loader = PyPDFLoader(Config.PDF_PATH)
    documents = loader.load()
    
    # Split into chunks (pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create Database
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=Config.DB_DIR)
    
    # Connect to Phi-3
    llm = Ollama(model=Config.MODEL_NAME)
    
    # Create the Q&A Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    print(f"{Colors.GREEN}✅ System Online & Ready.{Colors.ENDC}")
    return qa_chain

def main():
    # 1. Start the System
    qa_chain = initialize_system()
    
    # 2. Main Loop
    while True:
        print(f"\n{Colors.HEADER}{'='*40}{Colors.ENDC}")
        choice = input(f"Select: [1] Voice Mode  [2] Text Mode  [q] Quit\n> ")
        
        if choice == 'q':
            print("Shutting down...")
            break
        
        query_text = ""
        
        # VOICE MODE
        if choice == '1':
            try:
                audio_file = record_audio()
                query_text = transcribe_audio(audio_file)
                print(f"{Colors.CYAN}🗣️  You said: \"{query_text}\"{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}❌ Audio Error: {e}{Colors.ENDC}")
                continue
                
        # TEXT MODE
        elif choice == '2':
            query_text = input(f"{Colors.YELLOW}❓ Enter Question: {Colors.ENDC}")
            
        # PROCESS QUESTION
        if query_text.strip():
            print(f"{Colors.YELLOW}🧠 Thinking...{Colors.ENDC}")
            
            # Ask the AI
            response = qa_chain.invoke({"query": query_text})
            
            # Show Answer
            print(f"\n{Colors.CYAN}💡 Answer:\n{response['result']}\n{Colors.ENDC}")
            
            # Show Sources (Page Numbers)
            print(f"{Colors.BLUE}📄 Sources:{Colors.ENDC}")
            seen = set()
            for doc in response['source_documents']:
                page = doc.metadata.get('page', -1) + 1
                if page not in seen:
                    print(f" - Page {page}")
                    seen.add(page)

if __name__ == "__main__":
    main()