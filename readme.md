Here’s a **short and clear README** you can drop into your GitHub repo for this project:

---

# 📄 PDF Retrieval-Augmented Generation (RAG) System

This project is a **PDF-based RAG pipeline** built with **LangChain, Ollama, ChromaDB, and ElevenLabs**.
It lets you load PDFs, split them into chunks, attach metadata, store them in a **vector database**, and query them using a **multi-query retrieval strategy** — with optional **text-to-speech output**.

---

## 🚀 Features

* **PDF ingestion** via `PDFPlumberLoader`
* **Text chunking** with overlap using `RecursiveCharacterTextSplitter`
* **Metadata tagging** (title, author, date) for filtering and better retrieval
* **Vector storage** in [ChromaDB](https://docs.trychroma.com/)
* **Multi-query retriever** to improve recall
* **LLM querying** using [Ollama](https://ollama.ai/) (default: `llama3.2`)
* **Text-to-speech** output via [ElevenLabs](https://elevenlabs.io/)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sourav01112/voice-rag-pdf-assistant.git
cd voice-rag-pdf-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file for ElevenLabs API
echo "ELEVENLABS_API_KEY=your_api_key_here" > .env
```

---

## 📂 Project Structure

```
voice-rag-pdf-assistant/
│── data/                  # PDF files go here
│── db/vector_db/          # ChromaDB persistence
│── rag_system.py          # Main RAG pipeline
│── requirements.txt
│── README.md
```

---

## 🛠 Usage

```bash
python rag_system.py
```

* Place your `.pdf` files inside the `data/` folder.
* The system will:

  1. Load PDFs
  2. Split into text chunks
  3. Add metadata
  4. Store embeddings in ChromaDB
  5. Run a multi-query retrieval chain
  6. Answer your question
  7. Optionally convert the answer to speech

---

## 📌 Example Output

```text
Loading PDF files...
Processing PDF file: report.pdf
Pages loaded: 12
Splitting text into chunks...
Created 45 text chunks
Adding metadata...
Setting up vector database...
Vector database created and populated
Setting up retrieval chain...
Retrieval chain setup complete

Query: Does the document mention any specific technologies?
Response: Yes, it discusses blockchain and AI applications in the financial sector.

Converting to speech...
```

---

## 🔗 Requirements

* Python 3.9+
* [Ollama](https://ollama.ai/) installed locally
* ElevenLabs API key (optional for TTS)

---
