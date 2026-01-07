# TARK (Technical Aerospace Reasoning Kernel) 🛰️
**Secure, Offline Voice-Activated AI for Deep Space Operations**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/Tech-Offline%20RAG-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## Mission Overview
Deep space missions (e.g., Artemis, Mars) face communication latencies of up to **20 minutes**. In critical scenarios, astronauts cannot rely on Ground Control for immediate technical support.

**TARK** is an autonomous "Second Brain" designed for air-gapped environments. It ingests technical flight manuals locally and provides instant, voice-activated answers to flight parameter queries without internet connectivity.

## Project Origin & Solution
As the **Sole Architect and Lead Developer**, I conceptualized TARK to bridge the "Mars Latency" gap. 

- **The Challenge:** Traditional AI assistants rely on cloud processing, which is impossible in deep space. 
- **The Solution:** I engineered a local Retrieval-Augmented Generation (RAG) pipeline that runs 100% on edge hardware, ensuring zero data leakage and mission-critical reliability.

## Key Features
* **100% Offline Privacy:** Runs locally on edge hardware using quantized models (**Phi-3**).
* **Verifiable Accuracy:** Uses a RAG pipeline to cite specific page numbers for every answer (e.g., *Ref: p.24*).
* **Voice-to-Action:** Integrated "Push-to-Talk" interface using **OpenAI Whisper**.
* **Mission Control UI:** A high-contrast, professional dashboard built with **Streamlit**, designed for engineering environments.

##  System Architecture
1. **Ingestion:** Technical PDFs (like IDSS manuals) are parsed and split into chunks.
2. **Embedding:** Text is converted to vectors using `all-MiniLM-L6-v2` and stored in a local **ChromaDB**.
3. **Retrieval:** Whisper transcribes voice queries; the system performs a similarity search (`k=10`) to find precise technical data.
4. **Reasoning:** The local Phi-3 model processes the context to generate an accurate, cited response.

## Tech Stack
 * **Language:** Python 3.10+
 * **LLM:** Ollama (Phi-3 Mini)
 * **Vector DB:** ChromaDB
 * **Speech:** OpenAI Whisper
 * **Orchestration:** LangChain
 * **Interface:** Streamlit

## Installation

 1. **Clone the repository**
   bash
   git clone [https://github.com/haniiiyee/TARK-Aerospace-AI.git](https://github.com/haniiiyee/TARK-Aerospace-AI.git)
   cd TARK-Aerospace-AI

 2. **Install Dependencies**
    Ensure you have Python installed, then run the following to install the required libraries:
    ```bash
    pip install -r requirements.txt

 3. **Setup Ollama (Local AI Engine)**
    TARK requires Ollama to run the LLM locally.
    Download and install Ollama.
    Once installed, pull the Phi-3 model by running this command in your terminal:
    bash
    ollama pull phi3

 4. **Launch Mission Control**
    Start the TARK interface using Streamlit:
    Bash
    streamlit run app.py

**System Architecture**
 Ingestion: Technical PDFs (like IDSS manuals) are parsed and split into chunks.
 Embedding: Text is converted to vectors using all-MiniLM-L6-v2 and stored in a local ChromaDB.
 Retrieval: Whisper transcribes voice queries; the system performs a similarity search (k=10) to find precise technical data.
 Reasoning: The local Phi-3 model processes the context to generate an accurate, cited response.

Architected & Built by Hani Mohammad Kaif