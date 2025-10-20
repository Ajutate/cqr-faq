# CQR-FAQ System - Customer Query Response System

**Version:** 2.0  
**Last Updated:** October 20, 2025

A production-ready Retrieval-Augmented Generation (RAG) system for automated customer query response generation using FAQ knowledge bases. Features async processing, multi-format document support, and strict text extraction to prevent hallucination.

## 🚀 Key Features

### Core Functionality
- 📄 **Multi-Format Support**: Upload PDF, DOC, DOCX, and TXT documents
- 🏷️ **Document Categorization**: Organize documents with category and sub-category metadata
- 🔍 **Advanced RAG**: Semantic search using 768-dimensional embeddings (nomic-embed-text)
- 🤖 **Zero-Hallucination LLM**: Strict text extraction with temperature=0.0
- ⚡ **Async Processing**: Background task execution with immediate API response
- 📊 **Status Tracking**: Real-time processing status (InProgress/Completed/Failed)
- ⏱️ **Performance Metrics**: Processing time tracking and timestamps

### API Features
- 🔄 **RESTful API**: FastAPI with auto-generated Swagger UI and ReDoc
- 🔒 **Status-Aware Responses**: Handles concurrent requests with status checking
- 🐛 **Debug Endpoints**: Vector DB inspection and RAG query testing
- 📦 **Batch Upload**: Support for multiple document uploads with metadata

## 📋 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI 0.115.0+ | High-performance async API |
| **LLM** | Ollama (qwen3:latest) | Text extraction with temp=0.0 |
| **Embeddings** | nomic-embed-text | 768-dim semantic vectors |
| **Vector DB** | ChromaDB 1.1.1+ | Persistent vector storage |
| **Document Processing** | pdfplumber, python-docx | Multi-format extraction |
| **Text Splitting** | RecursiveCharacterTextSplitter | Context-preserving chunking |
| **Background Tasks** | FastAPI BackgroundTasks | Async request processing |

## 🔧 Prerequisites

### Required Software

1. **Python 3.10+** with conda/venv support
2. **Ollama** (local LLM runtime)
   - Download: https://ollama.ai
   - Required models:
     ```bash
     ollama pull qwen3:latest        # Main LLM
     ollama pull nomic-embed-text    # Embeddings
     ```
   - Verify installation:
     ```bash
     ollama list
     ```

### System Requirements
- **RAM**: 8GB minimum (16GB recommended for large documents)
- **Disk**: 10GB free space (for models and vector DB)
- **OS**: Windows 10/11, macOS, or Linux

## Installation

## 📥 Installation

### 1. Clone/Download Project
```bash
cd c:\WS\CQR-FAQ
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n cqr-faq python=3.12
conda activate cqr-faq

# Or using venv
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
Create `.env` file:
```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## 🎯 Quick Start

### 1. Start Ollama
```bash
ollama serve
```

### 2. Run Application
```bash
python main.py
```

Server starts at: **http://localhost:8000**

### 3. Access Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Production Endpoints

| Method | Endpoint | Description | Status Code |
|--------|----------|-------------|-------------|
| POST | `/upload` | Upload document with metadata | 200 OK |
| POST | `/cqr` | Process customer query (async) | 202 Accepted |
| POST | `/response` | Get response by case number | 200/202/404/500 |
| GET | `/responses/list` | List all case numbers | 200 OK |
| DELETE | `/response/{caseNumber}` | Delete stored response | 200 OK |

### Debug Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/debug/query` | Test RAG query with parameters |
| GET | `/debug/vector_chunks` | Inspect vector DB chunks |

### Auto-Generated Documentation

| Endpoint | Description |
|----------|-------------|
| GET `/docs` | Interactive Swagger UI |
| GET `/redoc` | ReDoc API documentation |

## 💼 Usage Examples

### 1. Upload FAQ Document with Category

**PowerShell:**
```powershell
$uri = "http://localhost:8000/upload"
$file = "C:\path\to\faq.pdf"
$form = @{
    file = Get-Item -Path $file
    category = "PMAY-U 2.0"
    sub_category = "Eligibility"
}
Invoke-RestMethod -Method Post -Uri $uri -Form $form
```

**curl:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@faq.pdf" \
  -F "category=PMAY-U 2.0" \
  -F "sub_category=Eligibility"
```

**Response:**
```json
{
  "message": "FAQ PDF processed successfully",
  "filename": "faq.pdf",
  "chunks": 25,
  "sections": 1,
  "category": "PMAY-U 2.0",
  "sub_category": "Eligibility"
}
```

### 2. Process Customer Query (Async)

**Request:**
```bash
curl -X POST "http://localhost:8000/cqr" \
  -H "Content-Type: application/json" \
  -d '{
    "caseNumber": "TEST-001",
    "category": "PMAY-U 2.0",
    "subCategory": "Eligibility",
    "custName": "John Doe",
    "fileNumber": "FILE001",
    "requestBody": "What is the eligibility criteria?"
  }'
```

**Response (Immediate - 202 Accepted):**
```json
{
  "message": "Request accepted and processing in background",
  "caseNumber": "TEST-001",
  "status": "InProgress",
  "requestTimestamp": "2025-10-20T10:30:00"
}
```

### 3. Check Status / Get Response

**Request:**
```bash
curl -X POST "http://localhost:8000/response" \
  -H "Content-Type: application/json" \
  -d '{"caseNumber": "TEST-001"}'
```

**While Processing (202 Accepted):**
```json
{
  "caseNumber": "TEST-001",
  "status": "InProgress",
  "message": "Request is still being processed. Please check again later."
}
```

**When Completed (200 OK):**
```json
{
  "caseNumber": "TEST-001",
  "responseBody": "Families belonging to EWS/LIG/MIG segments...",
  "category": "PMAY-U 2.0",
  "subCategory": "Eligibility",
  "custName": "John Doe",
  "fileNumber": "FILE001"
}
```

### 4. Inspect Vector Database

**Request:**
```bash
curl "http://localhost:8000/debug/vector_chunks?include_embeddings=true&limit=10"
```

**Response:**
```json
{
  "total_returned": 10,
  "items": [
    {
      "id": "uuid-here",
      "source": "faq.pdf",
      "page": 1,
      "chunk": 0,
      "category": "PMAY-U 2.0",
      "sub_category": "Eligibility",
      "content_preview": "What is PMAY-U 2.0?...",
      "embedding_preview": [0.0222, -0.0154, ...],
      "embedding_dim": 768
    }
  ]
}
```

## 🏗️ Architecture

### Async Processing Flow

```
Client                  Server                  Background Task
  │                       │                           │
  ├─ POST /cqr ─────────→│                           │
  │                       ├─ Mark InProgress         │
  │                       ├─ Add Background Task ────→│
  │←─ 202 Accepted ──────┤                           │
  │  (status: InProgress) │                           │
  │                       │                    ┌──────┴──────┐
  │                       │                    │ RAG Query   │
  │                       │                    │ LLM Extract │
  │                       │                    │ Store Result│
  │                       │                    └──────┬──────┘
  │                       │←─ Update Status ──────────┤
  │                       │   (Completed)             │
  │                       │                           │
  ├─ POST /response ─────→│                           │
  │  {caseNumber}         ├─ Check Status             │
  │←─ 200 OK ────────────┤   (Completed)             │
     {responseBody}       │                           │
```

### RAG Pipeline

```
Document Upload                Query Processing
     ↓                              ↓
PDF/DOC/TXT                   Customer Question
     ↓                              ↓
Extract Text                  Build Query String
     ↓                              ↓
Chunk (1000/200)              Embed Query (768-dim)
     ↓                              ↓
Generate Embeddings           Vector Search (k=8)
     ↓                              ↓
Store in ChromaDB             Retrieve Chunks
     ↓                              ↓
Persist to Disk               Build Context
                                   ↓
                              LLM Extract (temp=0.0)
                                   ↓
                              Pure Answer Text
                                   ↓
                              Store Response
```

### Storage Architecture

**Persistent (Survives Restart):**
- ✅ `./chroma_db/` - Vector embeddings and metadata
- ✅ `./uploads/` - Original uploaded files

**Temporary (Lost on Restart):**
- ⚠️ `processing_status` - Request status tracking
- ⚠️ `response_storage` - Generated responses

## 🔧 Configuration

### RAG Parameters (rag_service.py)

```python
# Text Chunking
chunk_size = 1000          # Characters per chunk
chunk_overlap = 200        # Overlap between chunks

# Vector Search
k = 8                      # Number of chunks to retrieve
score_threshold = 0.3      # Removed (using top-k instead)

# LLM Settings (main.py)
temperature = 0.0          # Deterministic (no creativity)
num_predict = 2048         # Max output tokens
```

### Supported Document Types

| Format | Library | Notes |
|--------|---------|-------|
| PDF | pdfplumber | Tables + text extraction |
| DOC/DOCX | python-docx | Paragraph extraction |
| TXT | Built-in | Plain text |

## 🧪 Testing

### Run Vector DB Inspection
```bash
python inspect_chromadb.py
```

### Test Async CQR Flow
```bash
python test_async_cqr.py
```

### Test Vector Chunks Endpoint
```bash
python test_vector_chunks.py
```

## 🐛 Troubleshooting

### Issue: "Empty string" responses

**Cause:** Similarity threshold filtering out correct chunks  
**Solution:** Using top-k retrieval (no threshold) + strict LLM filtering

### Issue: LLM adding extra information

**Cause:** Temperature > 0 or combining multiple chunks  
**Solution:** Set `temperature=0.0` and use STRICT extraction prompt

### Issue: "InProgress" status stuck

**Cause:** Background task failed silently  
**Solution:** Check server logs, verify Ollama is running

### Issue: Slow processing

**First query:** ~10-15 seconds (includes embedding generation)  
**Subsequent:** ~4-6 seconds  
**Large PDFs:** May take longer for initial embedding

### Issue: Ollama connection error

```bash
# Check Ollama is running
ollama list

# Restart Ollama
ollama serve

# Verify models
ollama pull qwen3:latest
ollama pull nomic-embed-text
```

## 📊 Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Upload (25 chunks) | ~2-3s | First-time embedding |
| Query Processing | ~4-6s | Includes LLM generation |
| Vector Search | ~50-100ms | ChromaDB similarity search |
| Status Check | <10ms | In-memory lookup |

## 🚀 Production Deployment

### Recommended Changes

1. **Replace In-Memory Storage:**
   - Use PostgreSQL/MongoDB for `response_storage`
   - Persist `processing_status` to database

2. **Add Authentication:**
   - API key validation
   - JWT tokens for user sessions

3. **Add Rate Limiting:**
   - Throttle requests per IP
   - Queue management for background tasks

4. **Scale Vector DB:**
   - Consider Pinecone/Weaviate for production
   - Or self-hosted Milvus/Qdrant

5. **Add Monitoring:**
   - Prometheus metrics
   - Error tracking (Sentry)
   - Performance monitoring (New Relic)

## 📁 Project Structure

```
CQR-FAQ/
├── main.py                          # FastAPI application
├── rag_service.py                   # RAG logic & vector store
├── pdf_extractor.py                 # Document extraction
├── requirements.txt                 # Python dependencies
├── .env                             # Environment config (optional)
├── README.md                        # This file
├── ARCHITECTURE.md                  # System architecture
├── ASYNC_CQR_IMPLEMENTATION.md      # Async docs
├── REQUIREMENTS_ANALYSIS.md         # Dependency analysis
│
├── chroma_db/                       # Vector database (persistent)
│   ├── chroma.sqlite3
│   └── collection_id/
│
├── uploads/                         # Uploaded files (persistent)
│
└── test_*.py                        # Test scripts
```

## 📝 API Request Models

### CQRRequest
```python
{
  "caseNumber": str,      # Required - Unique case ID
  "category": str,        # Required - Main category
  "subCategory": str,     # Required - Sub-category
  "custName": str,        # Required - Customer name
  "fileNumber": str,      # Required - File reference
  "requestBody": str      # Required - Customer question
}
```

### CQRResponse
```python
{
  "caseNumber": str,      # Case ID
  "responseBody": str     # Generated answer
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Database integration for persistent storage
- User authentication
- Rate limiting
- Advanced filtering by category
- Hybrid search (BM25 + semantic)

## 📄 License

MIT License - Free to use and modify

## 📞 Support

- Check `/docs` endpoint for interactive API documentation
- Review `ARCHITECTURE.md` for system design details
- Run debug scripts in project root for troubleshooting

## 🔄 Version History

### v2.0 (October 2025)
- ✅ Async processing with background tasks
- ✅ Status tracking (InProgress/Completed/Failed)
- ✅ Multi-format support (PDF/DOC/DOCX/TXT)
- ✅ Category/subcategory metadata
- ✅ Strict text extraction (temp=0.0)
- ✅ Debug endpoints for vector DB inspection
- ✅ Processing time tracking
- ✅ Enhanced error handling

### v1.0 (Initial Release)
- Basic RAG functionality
- PDF upload and processing
- Synchronous query processing

---

**Built with ❤️ using FastAPI, Ollama, and ChromaDB**
