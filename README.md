# Enterprise RAG Chatbot for Cybersecurity & Privacy

An Enterprise Retrieval-Augmented Generation (RAG) chatbot designed to answer cybersecurity and information-security questions using a NIST knowledge base.

The system combines semantic search, keyword-based retrieval, Reciprocal Rank Fusion (RRF), query routing, grounded LLM generation, source citation, and retrieval monitoring to produce document-grounded answers.

---

## 📌 Project Overview

This project implements an advanced RAG pipeline for a cybersecurity and privacy knowledge base.

The primary source document used in the current implementation is:

**NIST SP 800-100 — Information Security Handbook: A Guide for Managers**

The document contains cybersecurity and information-security guidance published by the National Institute of Standards and Technology (NIST).

The system retrieves relevant document chunks before generating an answer. The LLM is instructed to answer only from the retrieved context and avoid unsupported information.

---

## 🎯 Objectives

- Build an enterprise-oriented RAG chatbot
- Retrieve relevant information from cybersecurity documents
- Combine semantic and keyword-based search
- Improve retrieval using Reciprocal Rank Fusion
- Route queries intelligently
- Generate grounded answers using an LLM
- Reduce hallucinations through strict context grounding
- Provide source and chunk-level citations
- Monitor retrieval performance
- Evaluate the retrieval pipeline using a benchmark dataset
- Provide an interactive Streamlit interface

---

## 🚀 Key Features

### 1. Document Processing

- PDF-based knowledge source
- Text extraction
- Text cleaning
- Chunk creation
- Page and chunk metadata
- Source mapping

### 2. Semantic Retrieval

The project uses:

**BAAI/bge-small-en-v1.5**

The embedding model is executed using ONNX Runtime.

Embeddings are normalized and indexed using FAISS.

### 3. Keyword Retrieval

The system uses:

**BM25**

BM25 provides lexical matching and helps retrieve documents containing important keywords and phrases.

### 4. Hybrid Retrieval

The system combines:

- FAISS semantic retrieval
- BM25 keyword retrieval

The two retrieval results are combined using:

**Reciprocal Rank Fusion (RRF)**

This allows the system to benefit from both semantic similarity and exact keyword matching.

### 5. Query Expansion

Important cybersecurity queries can be expanded with related terminology to improve retrieval coverage.

Examples include:

- Information security
- Security awareness
- Governance
- Risk management

### 6. Intelligent Query Router

Queries are classified into different routes:

- `DOCUMENT_RETRIEVAL`
- `CONVERSATION_HISTORY`
- `CLARIFICATION`
- `OUTSIDE_KNOWLEDGE_BASE`

This allows the application to determine how a question should be handled before retrieval and answer generation.

### 7. Grounded LLM Answer Generation

The application uses Google Gemini for answer generation.

The LLM receives the retrieved document context and is instructed to:

- Answer only from the provided context
- Avoid assumptions
- Avoid adding unsupported knowledge
- Provide source information
- Return a fallback response when the information is unavailable

Fallback response:

> The information is not available in the provided documents.

### 8. Source Citation

Answers can identify the document chunks used to generate the response.

Example:

```text
Source: Chunk 536