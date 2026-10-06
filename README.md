# Company RAG Assistant

A local **Retrieval-Augmented Generation (RAG)** application that answers questions from company policy data.

The project uses:

- **Sentence Transformers** for embeddings
- **ChromaDB** as the vector database
- **Semantic Search** for retrieving relevant information
- **Ollama + Qwen3:1.7b** as the local LLM
- **Streamlit** for the web interface

The complete system runs locally, so no paid LLM API is required.

---

## Project Overview

Traditional LLMs may not know private company policies and can sometimes generate incorrect information.

This project solves that problem using **RAG — Retrieval-Augmented Generation**.

The application:

1. Loads company policy data from a CSV file.
2. Cleans the text.
3. Splits policy documents into smaller chunks.
4. Converts chunks into embeddings.
5. Stores embeddings in ChromaDB.
6. Accepts a user question.
7. Converts the question into an embedding.
8. Performs semantic search.
9. Retrieves the most relevant policy chunks.
10. Sends the retrieved context and question to Qwen using Ollama.
11. Generates the final answer.

---

## RAG Architecture

```text
Company Policy CSV
        |
        v
Text Preprocessing
        |
        v
Chunking
        |
        v
SentenceTransformer
        |
        v
Embeddings
        |
        v
ChromaDB
        |
        v

User Question
        |
        v
Query Embedding
        |
        v
Semantic Search
        |
        v
Relevant Policy Chunks
        |
        v
Prompt + Retrieved Context
        |
        v
Ollama
        |
        v
Qwen3:1.7b
        |
        v
Final Answer
```

---

## Dataset

The project uses:

```text
company_rag_dataset.csv
```

Dataset columns:

| Column | Description |
|---|---|
| id | Unique policy ID |
| category | Policy category |
| title | Policy title |
| content | Policy text |
| owner | Policy owner |

The dataset contains approximately **120 records**.

After chunking, approximately **240 chunks** are stored in ChromaDB.

---

## Technologies Used

- Python 3.12
- Pandas
- Sentence Transformers
- ChromaDB
- Ollama
- Qwen3:1.7b
- Streamlit

---

## Project Structure

```text
company_rag_assistant/
│
├── company_rag_dataset.csv
├── main.py
├── app.py
├── requirements.txt
├── README.md
└── .venv/
```

### main.py

Contains the main RAG pipeline.

### app.py

Contains the Streamlit web application.

### company_rag_dataset.csv

Contains company policy data.

### requirements.txt

Contains Python dependencies.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Move into the project folder:

```bash
cd company_rag_assistant
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Install Ollama on your computer.

Download the Qwen model:

```bash
ollama pull qwen3:1.7b
```

Check installed models:

```bash
ollama list
```

Test Qwen:

```bash
ollama run qwen3:1.7b
```

---

## Run the Project

Make sure Ollama is running.

Run the Python RAG application:

```bash
python main.py
```

For the Streamlit application:

```bash
streamlit run app.py
```

Streamlit normally opens:

```text
http://localhost:8501
```

---

## Example Question

```text
How many annual leave days do full-time employees receive?
```

The RAG system performs semantic search and retrieves the relevant company policy.

Example output:

```text
Full-time employees receive 20 days of annual leave per calendar year.
```

---

## How Semantic Search Works

Normal keyword search searches mainly for exact words.

Semantic search searches based on **meaning**.

Example user question:

```text
How much vacation do employees get?
```

The company document may contain:

```text
Full-time employees receive 20 days of annual leave per calendar year.
```

The words **vacation** and **annual leave** are different, but their meanings are similar.

Sentence Transformers convert both the user question and company policy chunks into vectors.

ChromaDB compares these vectors and retrieves the most semantically similar chunks.

---

## What Are Embeddings?

Embeddings are numerical vector representations of text.

Example:

```text
"annual leave"
        ↓
Embedding Model
        ↓
[0.12, -0.45, 0.78, ...]
```

Texts with similar meanings usually have vectors that are close to each other in vector space.

---

## Why ChromaDB?

ChromaDB is used to store and search embeddings.

Instead of searching through every company policy manually, ChromaDB quickly finds the chunks whose embeddings are closest to the user's query embedding.

---

## Why RAG?

RAG combines two processes:

### Retrieval

Retrieve relevant information from company documents.

### Generation

Give that information to an LLM so it can generate a natural-language response.

Flow:

```text
User Question
      ↓
Retrieve Relevant Company Data
      ↓
Give Context to LLM
      ↓
Generate Answer
```

This helps reduce hallucinations because the LLM receives relevant company information before answering.

---

## Current Features

- CSV dataset loading
- Missing-value handling
- Text cleaning
- Document chunking
- Sentence embeddings
- ChromaDB vector storage
- Semantic search
- Context retrieval
- Local Qwen generation
- Ollama integration
- Streamlit web interface
- No paid LLM API required

---

## Future Improvements

The project can be upgraded with:

- Better chunking strategies
- Metadata filtering
- Hybrid search
- BM25 keyword search
- Dense semantic search
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Retrieval evaluation
- RAG evaluation
- Source citations
- Chat history
- FastAPI backend
- PostgreSQL
- Docker
- Authentication
- Logging
- Monitoring
- Cloud deployment
- CI/CD

---

## Learning Concepts Covered

This project demonstrates:

- Text preprocessing
- Chunking
- Embeddings
- Vector databases
- Cosine similarity
- Semantic search
- Retrieval
- Prompt construction
- Local LLM inference
- Ollama
- ChromaDB
- Sentence Transformers
- Retrieval-Augmented Generation
- Streamlit

---

## Author

**Tharun Kumar Reddy**

Generative AI / RAG Project