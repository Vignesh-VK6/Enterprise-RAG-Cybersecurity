# Enterprise RAG Chatbot — System Architecture

## 1. System Overview

The Enterprise RAG Chatbot is a Retrieval-Augmented Generation system designed to answer cybersecurity and information-security questions using a controlled knowledge base.

The current knowledge base contains the NIST SP 800-100 Information Security Handbook.

The system combines semantic retrieval, keyword retrieval, query routing, grounded answer generation, source citation, and retrieval monitoring.

---

## 2. High-Level Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
Intelligent Query Router
  |
  v
Query Processing / Expansion
  |
  v
+-----------------------------+
|       Hybrid Retrieval      |
|                             |
|  FAISS        BM25          |
|  Semantic     Keyword       |
|  Search       Search        |
|      \         /             |
|       \       /              |
|        RRF Fusion            |
|             |                |
|             v                |
|       Ranking / Scoring      |
+-------------|---------------+
              |
              v
       Top Relevant Chunks
              |
              v
       Context Construction
              |
              v
       Grounded Gemini LLM
              |
              v
        Source Citation
              |
              v
         Final Answer
              |
              v
          Monitoring
```

---

## 3. Knowledge Base

The current knowledge base is based on:

**Document:** NIST SP 800-100 Information Security Handbook: A Guide for Managers

The document belongs to the cybersecurity and information-security domain.

The processed document currently contains:

* 1 source PDF
* 178 pages
* 630 text chunks

The chunks are stored and aligned with the FAISS vector index.

---

## 4. Document Processing Pipeline

The document processing pipeline converts the source PDF into searchable text chunks.

```text
NIST PDF
   |
   v
Text Extraction
   |
   v
Text Cleaning
   |
   v
Text Chunking
   |
   v
630 Chunks
   |
   v
Embedding Generation
   |
   v
FAISS Index
```

The processed chunks are used as the knowledge source for retrieval.

---

## 5. Embedding Generation

The project uses:

**Embedding Model:** BAAI/bge-small-en-v1.5

The embedding model is executed using ONNX Runtime.

The embedding process is:

```text
Text Chunk
    |
    v
Tokenizer
    |
    v
BGE-small-en-v1.5
    |
    v
ONNX Runtime
    |
    v
Mean Pooling
    |
    v
Vector Normalization
    |
    v
FAISS
```

The generated embeddings are normalized before being used for similarity search.

---

## 6. Vector Database

The project uses **FAISS** for vector similarity search.

Current vector index:

```text
vector_db/onnx_faiss.index
```

The FAISS index contains vectors corresponding to the processed document chunks.

FAISS is used to identify document chunks that are semantically similar to the user's question.

---

## 7. BM25 Keyword Retrieval

The system also uses BM25 for keyword-based retrieval.

BM25 is useful when the query contains important technical terms or exact keywords.

The BM25 process is:

```text
User Query
    |
    v
Tokenization
    |
    v
BM25 Search
    |
    v
Candidate Chunks
```

BM25 is built over the complete 630-chunk corpus.

---

## 8. Hybrid Retrieval

The system combines semantic retrieval and keyword retrieval.

### Semantic Retrieval

FAISS uses BGE embeddings to find semantically similar chunks.

```text
Query
  |
  v
BGE Embedding
  |
  v
FAISS Similarity Search
  |
  v
Semantic Candidates
```

### Keyword Retrieval

BM25 identifies chunks based on keyword relevance.

```text
Query
  |
  v
Tokenization
  |
  v
BM25 Search
  |
  v
Keyword Candidates
```

### Result Fusion

The FAISS and BM25 results are combined using Reciprocal Rank Fusion (RRF).

```text
              User Query
                  |
        +---------+---------+
        |                   |
        v                   v
      FAISS               BM25
   Semantic Search     Keyword Search
        |                   |
        +---------+---------+
                  |
                  v
              RRF Fusion
                  |
                  v
          Ranking / Scoring
                  |
                  v
           Top 3 Chunks
```

This approach allows the system to use both semantic similarity and exact keyword matching.

---

## 9. Query Expansion

The system supports query expansion for selected cybersecurity topics.

For example, a query related to information security may be expanded using related terms such as:

* protect information
* protect information systems
* confidentiality
* integrity
* availability
* security controls
* security requirements

Similar expansion rules are available for topics such as:

* Information Security
* Security Awareness
* Governance
* Risk Management

Query expansion is applied before retrieval when a matching topic is detected.

---

## 10. Intelligent Query Router

The query router determines how a user query should be handled.

The system supports four routes:

```text
DOCUMENT_RETRIEVAL
CONVERSATION_HISTORY
CLARIFICATION
OUTSIDE_KNOWLEDGE_BASE
```

### DOCUMENT_RETRIEVAL

Used when the query is related to the cybersecurity knowledge base.

Example:

```text
What is information security governance?
```

### CONVERSATION_HISTORY

Used when the current query depends on previous conversation context.

Example:

```text
What are its objectives?
```

### CLARIFICATION

Used when the user's query is ambiguous and requires clarification.

Example:

```text
Explain that.
```

### OUTSIDE_KNOWLEDGE_BASE

Used when the question is outside the available knowledge base.

Example:

```text
What is the weather today?
```

---

## 11. Context Construction

After retrieval, the highest-ranked chunks are selected and prepared as context for the language model.

```text
User Query
    |
    v
Hybrid Retrieval
    |
    v
Candidate Chunks
    |
    v
Ranking
    |
    v
Top Relevant Chunks
    |
    v
Context Construction
    |
    v
Grounded LLM
```

The retrieved context provides the evidence used for answer generation.

---

## 12. Grounded Answer Generation

The system uses Gemini for answer generation.

The LLM receives:

1. The user question
2. Retrieved document context
3. Grounding instructions

The grounding instructions require the model to:

* Use only the retrieved document context.
* Avoid unsupported assumptions.
* Avoid adding outside knowledge.
* Generate answers based on retrieved evidence.
* Return a fallback response when sufficient information is unavailable.

Fallback response:

```text
The information is not available in the provided documents.
```

---

## 13. Hallucination Control

The system reduces unsupported answers through retrieval-grounded generation.

```text
User Question
      |
      v
Retrieve Evidence
      |
      v
Construct Context
      |
      v
Grounded Prompt
      |
      v
Gemini
      |
      v
Evidence-Based Answer
```

If the retrieved context does not contain sufficient information, the system uses the predefined fallback response.

This prevents the application from intentionally generating answers from unrelated external knowledge.

---

## 14. Source Citation

The system tracks the source document and chunk associated with retrieved evidence.

Example:

```text
Source: pdf 1.pdf
Chunk: 33
```

The Streamlit interface can display:

* Source document
* Chunk ID
* FAISS score
* BM25 score
* Retrieved context

This provides traceability between the generated response and the retrieved document evidence.

---

## 15. Monitoring

Retrieval monitoring is implemented using:

```text
src/monitoring.py
```

Retrieval logs are stored in:

```text
logs/retrieval_logs.jsonl
```

The monitoring system records information such as:

* User query
* Retrieved documents
* Retrieval scores
* Response time
* Errors

These logs can be used for debugging, performance analysis, and future optimization.

---

## 16. Evaluation

A benchmark containing 50 NIST-related questions was created for retrieval evaluation.

Current retrieval benchmark results:

| Metric                    |     Result |
| ------------------------- | ---------: |
| Total Questions           |         50 |
| Successful Queries        |         50 |
| Errors                    |          0 |
| Document Retrieval Routes |         30 |
| Outside Knowledge Routes  |         20 |
| Top-1 Retrieval Success   |        46% |
| Top-3 Retrieval Success   |        58% |
| Average Top-3 Relevance   |     24.63% |
| Average Retrieval Latency | 0.0065 sec |

These results evaluate the retrieval layer.

They should not be interpreted as overall end-to-end answer accuracy because the benchmark primarily measures retrieval performance.

---

## 17. Streamlit Application

The main application interface is implemented using Streamlit.

Main entry point:

```text
app.py
```

The application provides:

* User question input
* Generated answer
* Query route
* Response time
* Source information
* FAISS score
* BM25 score
* Retrieved context
* Conversation history

---

## 18. Main Project Components

```text
app.py
|
+-- Streamlit User Interface
|
src/
|
+-- rag_service.py
|      Main RAG orchestration
|
+-- query_router.py
|      Intelligent query routing
|
+-- llm_answer_generation.py
|      Grounded Gemini answer generation
|
+-- context_construction.py
|      Retrieved context preparation
|
+-- conversational_rag.py
|      Conversational retrieval
|
+-- multi_turn_rag.py
|      Multi-turn query handling
|
+-- hallucination_control.py
|      Grounding and fallback logic
|
+-- source_citation.py
|      Source mapping
|
+-- monitoring.py
|      Retrieval monitoring
|
+-- rag_evaluation.py
       Evaluation utilities

vector_db/
|
+-- onnx_faiss.index

benchmark/
|
+-- benchmark_questions.json
+-- benchmark_results.csv
+-- benchmark_runner.py
+-- benchmark_summary.json

logs/
|
+-- retrieval_logs.jsonl
```

---

## 19. End-to-End Request Flow

A typical request follows this sequence:

```text
1. User enters a question
          |
          v
2. Streamlit receives the query
          |
          v
3. Query Router classifies the query
          |
          v
4. Query expansion is applied when applicable
          |
          v
5. FAISS performs semantic retrieval
          |
          v
6. BM25 performs keyword retrieval
          |
          v
7. RRF combines retrieval results
          |
          v
8. Ranking/scoring selects relevant chunks
          |
          v
9. Context is constructed
          |
          v
10. Gemini receives grounded context
          |
          v
11. Answer is generated
          |
          v
12. Source chunks are attached
          |
          v
13. Response is displayed
          |
          v
14. Retrieval information is logged
```

---

## 20. Why RAG Is Used

A conventional chatbot may generate an answer based on the language model's learned knowledge.

This project instead retrieves relevant information from a controlled document collection before generating an answer.

```text
User Question
      |
      v
Retrieve Enterprise Knowledge
      |
      v
Retrieved Evidence
      |
      v
Grounded LLM
      |
      v
Final Answer
```

This architecture allows the system to answer using the selected enterprise knowledge base and provide source information for retrieved evidence.

---

## 21. Scalability

The current prototype contains 630 document chunks.

For a larger enterprise deployment, the architecture can be extended with:

* Persistent vector databases
* Metadata filtering
* Document-level access control
* Batch indexing pipelines
* Embedding services
* Caching
* Parallel retrieval
* Advanced reranking models
* Multiple document collections
* Monitoring dashboards
* Automated evaluation pipelines

For very large document collections, retrieval and storage infrastructure can be scaled independently from the user interface and LLM components.

---

## 22. Security Considerations

A production deployment should consider:

* Authentication
* Authorization
* Role-based document access
* API key protection
* Secret management
* Input validation
* Prompt injection protection
* Sensitive document handling
* Audit logging
* Data encryption
* Access-controlled vector indexes

API keys must not be committed to GitHub.

Secrets should be stored using environment variables or an appropriate secret-management system.

---

## 23. Current Project Status

The project currently includes:

* Document processing
* 630 document chunks
* BGE-small-en-v1.5 embeddings
* ONNX Runtime inference
* FAISS vector search
* BM25 keyword search
* Hybrid retrieval
* RRF-based result fusion
* Query expansion
* Intelligent query routing
* Conversational retrieval
* Grounded Gemini generation
* Hallucination control
* Source citation
* Retrieval monitoring
* Benchmark evaluation
* Streamlit user interface
* Project README documentation
* System architecture documentation

The system is currently structured as a working prototype suitable for demonstration, evaluation, and further production-oriented development.
