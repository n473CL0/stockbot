import os
import socket
import subprocess
import time
import streamlit as st
from langchain_ollama import OllamaLLM

def is_port_in_use(port=11434):
    """Check if Ollama server is already running."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_ollama():
    """Start Ollama server in background if not running."""
    if not is_port_in_use():
        st.info("⚙️ Starting Ollama server...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

def run_ollama(model_name="mistral:7b"):
    """Ensure Ollama is running and return an OllamaLLM instance."""
    os.environ["OLLAMA_INTEL_GPU"] = "false"
    os.environ["OLLAMA_NUM_PARALLEL"] = "1"

    start_ollama()

    try:
        t0 = time.time()
        llm = OllamaLLM(model=model_name)
        _ = llm.invoke("ping")  # quick sanity check
        elapsed = time.time() - t0
        st.success(f"✅ Model {model_name} ready in {elapsed:.2f}s")
        return llm
    except Exception as e:
        st.error(f"❌ Could not connect to Ollama: {e}")
        raise
