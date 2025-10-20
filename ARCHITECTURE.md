# CQR-FAQ System Architecture

**Last Updated:** October 20, 2025  
**Version:** 2.0 - Async Processing with Background Tasks

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CQR-FAQ SYSTEM                                │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  STEP 1: Setup FAQ Knowledge Base (One-time)                         │
└──────────────────────────────────────────────────────────────────────┘

    📄 Document File (PDF/DOC/DOCX/TXT)
         │                             
         │ POST /upload                
         │ + category (optional)
         │ + sub_category (optional)
         ↓                             
    ┌──────────────┐                   
    │ Doc Extractor│                   
    └──────┬───────┘                   
           │ Extract & Split            
           ↓                            
    ┌─────────────────┐                
    │ Text Chunking   │                
    │ (1000 chars)    │                
    │ (200 overlap)   │                
    └────────┬────────┘                
             │                          
             │ Generate Embeddings (768-dim)
             │ + Add metadata (category, sub_category)
             ↓                          
    ┌────────────────────┐             
    │ ChromaDB           │             
    │ (Vector Database)  │ ← Persists on disk
    │ ./chroma_db/       │             
    │ nomic-embed-text   │
    └────────────────────┘             


┌──────────────────────────────────────────────────────────────────────┐
│  STEP 2: Process Customer Query (POST /cqr) - ASYNC                  │
└──────────────────────────────────────────────────────────────────────┘

    📨 Customer Request                
    {                                  
      caseNumber: "CAS001",            
      category: "PMAY-U 2.0",          
      subCategory: "Eligibility",      
      custName: "John Doe",            
      fileNumber: "FILE001",
      requestBody: "What is eligibility?"
    }                                  
         │                             
         │ POST /cqr                   
         ↓                             
    ┌──────────────────────────────┐   
    │ CQR Endpoint                 │   
    │ (Async with BackgroundTasks) │   
    └────────┬─────────────────────┘   
             │                         
             │ 1. Mark as "InProgress"
             │ 2. Return 202 Accepted (IMMEDIATELY)
             ↓                         
    Client ← {                         
      status: "InProgress",            
      caseNumber: "CAS001",            
      message: "Processing..."         
    }                                  
             │                         
             │ 3. Process in Background
             ↓                         
    ┌─────────────────────────────┐    
    │ Background Processing       │    
    │ (Non-blocking)              │    
    └────────┬────────────────────┘    
             │                         
             │ 4. Build Query          
             ↓                         
    "Category: PMAY-U 2.0,             
     Sub-Category: Eligibility,        
     Question: What is eligibility?"   
             │                         
             │ 5. Vector Search (k=8, threshold=0.3)
             ↓                         
    ┌────────────────────┐             
    │ ChromaDB           │             
    │ Query FAQ KB       │             
    │ (Similarity Search)│
    └────────┬───────────┘             
             │                         
             │ 6. Retrieve Top 8 Chunks
             ↓                         
    � FAQ Context:                    
    "Families belonging to EWS/LIG..." 
             │                         
             │ 7. Build STRICT Prompt (temp=0.0)
             ↓                         
    ┌────────────────────┐             
    │ Ollama LLM         │             
    │ (qwen3:latest)     │             
    │ temp=0.0           │
    │ num_predict=2048   │
    └────────┬───────────┘             
             │                         
             │ 8. Extract EXACT Answer 
             ↓                         
    📧 Response Text (pure answer)     
             │                         
             │ 9. Store with timestamp & timing
             ↓                         
    ┌────────────────────────────┐     
    │ In-Memory Dictionaries     │     
    │                            │
    │ processing_status[CAS001]  │
    │   = "Completed"            │
    │                            │     
    │ response_storage[CAS001] = {│     
    │   responseBody: "...",     │     
    │   category: "...",         │     
    │   processingTimeSeconds: 4.2│
    │   requestTimestamp: "...", │
    │   responseTimestamp: "..." │
    │ }                          │  ← Lost on restart
    └────────────────────────────┘                                  


┌──────────────────────────────────────────────────────────────────────┐
│  STEP 3: Retrieve Response (POST /response) - STATUS-AWARE           │
└──────────────────────────────────────────────────────────────────────┘

    🔍 POST /response                  
    { caseNumber: "CAS001" }           
         │                             
         │ 1. Check Status First
         ↓                             
    ┌────────────────────────┐         
    │ processing_status      │         
    │ [CAS001]               │         
    └────────┬───────────────┘         
             │                         
             ├─ "InProgress" → Return 202
             │  {                      
             │    status: "InProgress",
             │    message: "Still processing..."
             │  }                      
             │                         
             ├─ "Failed" → Return 500  
             │  {                      
             │    error: "Processing failed"
             │  }                      
             │                         
             └─ "Completed" → Continue 
                      ↓                
    ┌────────────────────────┐         
    │ response_storage       │         
    │ [CAS001]               │         
    └────────┬───────────────┘         
             │                         
             │ 2. Return Complete Response
             ↓                         
    {                                  
      caseNumber: "CAS001",            
      responseBody: "Families belonging...",
      category: "PMAY-U 2.0",          
      subCategory: "Eligibility",      
      custName: "John Doe",            
      fileNumber: "FILE001"            
    }                                  
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN.PY                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
  ┌──────────────────────────────────────────────────────────┐  │
  │  In-Memory Storage (Dictionaries)                         │  │
  │                                                            │  │
  │  processing_status: Dict[str, str]                        │  │
  │  • Tracks request status                                  │  │
  │  • Values: "InProgress", "Completed", "Failed"            │  │
  │                                                            │  │
  │  response_storage: Dict[str, Dict]                        │  │
  │  • Stores generated responses                             │  │
  │  • Key: caseNumber                                        │  │
  │  • Value: Complete response data + timestamps + timing    │  │
  │  • ⚠️  Lost on restart                                    │  │
  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
  ┌──────────────────────────────────────────────────────────┐  │
  │  Ollama LLM (qwen3:latest)                                │  │
  │  • Temperature: 0.0 (deterministic, no creativity)        │  │
  │  • num_predict: 2048 (longer responses)                   │  │
  │  • STRICT text extraction (no hallucination)              │  │
  │  • Uses FAQ context from RAG                              │  │
  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
  ┌──────────────────────────────────────────────────────────┐  │
  │  API Endpoints                                            │  │
  │  • POST   /upload                - Upload document       │  │
  │  • POST   /cqr                   - Process query (async) │  │
  │  • POST   /response              - Get stored response   │  │
  │  • GET    /responses/list        - List all cases        │  │
  │  • DELETE /response/{id}         - Delete response       │  │
  │  • POST   /debug/query           - Test RAG query        │  │
  │  • GET    /debug/vector_chunks   - Inspect vector DB     │  │
  │  • GET    /docs                  - Swagger UI            │  │
  │  • GET    /redoc                 - ReDoc API docs        │  │
  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ imports
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RAG_SERVICE.PY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RAGService Class                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ChromaDB (Vector Database)                               │  │
│  │  • Location: ./chroma_db/                                 │  │
│  │  • Stores FAQ embeddings                                  │  │
│  │  • ✅ Persists on disk                                    │  │
│  │  • Collection: pdf_documents                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Ollama Embeddings (nomic-embed-text)                     │  │
│  │  • Converts text to vectors                               │  │
│  │  • Used for FAQ storage & search                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Text Splitter                                            │  │
│  │  • RecursiveCharacterTextSplitter                         │  │
│  │  • chunk_size: 1000 chars                                 │  │
│  │  • chunk_overlap: 200 chars                               │  │
│  │  • Preserves context across chunks                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Methods                                                  │  │
│  │  • process_pdf()       - Store FAQ PDF in vector DB      │  │
│  │  • process_document()  - Store PDF/DOC/DOCX/TXT + metadata│ │
│  │  • query()             - Retrieve relevant FAQs (RAG)    │  │
│  │  • list_documents()    - List all sources in DB          │  │
│  │  • clear_database()    - Clear all vectors               │  │
│  │  • delete_document()   - Delete specific source          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ imports
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PDF_EXTRACTOR.PY                              │
├─────────────────────────────────────────────────────────────────┤
│  • Extracts text from PDF files                                 │
│  • Uses pdfplumber (MIT License)                                │
│  • Handles tables and complex layouts                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow - Complete Example

```
1. UPLOAD FAQ PDF
   ─────────────────────────────────────────────────
   Client → POST /upload (home_loan_faq.pdf)
          ↓
   PDF Extractor → Extract text & tables
          ↓
   Text Splitter → 150 chunks (500 chars each)
          ↓
   Embeddings → Convert to vectors
          ↓
   ChromaDB → Store in ./chroma_db/ ✅ PERSISTENT


2. CUSTOMER QUERY (ASYNC)
   ─────────────────────────────────────────────────
   Client → POST /cqr
   {
     caseNumber: "CAS001",
     category: "Interest Rate",
     requestBody: "What is the current ROI?"
   }
          ↓
   RAG Query → "Category: Interest Rate, Question: What is ROI?"
          ↓
   ChromaDB → Vector search → Top 5 relevant chunks
          ↓
   FAQ Context: "Current ROI is 8.5%..."
          ↓
   Ollama LLM → Generate email with FAQ context
          ↓
   Email Response: "Dear John, Current ROI is 8.5%..."
          ↓
   In-Memory Dict → response_storage["CAS001"] = {...}
          ↓
   Client ← { message: "Request processed", status: "stored" }
          ⚠️  NO RESPONSE BODY RETURNED


3. RETRIEVE RESPONSE
   ─────────────────────────────────────────────────
   Client → GET /response/CAS001
          ↓
   In-Memory Dict → Lookup response_storage["CAS001"]
          ↓
   Client ← {
     caseNumber: "CAS001",
     responseBody: "Dear John, Current ROI is 8.5%...",
     category: "Interest Rate",
     custName: "John Doe",
     ...
   }
```

---

## Storage Persistence

```
┌─────────────────────────────────────────────────────────────┐
│  PERSISTENT STORAGE (Survives Restart)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📂 ./chroma_db/                                            │
│     ├── chroma.sqlite3                                      │
│     └── 77aa3984-06da-496d-b536-952170a661f9/               │
│         └── [vector embeddings]                             │
│                                                              │
│  ✅ FAQ Knowledge Base                                      │
│  ✅ Vector embeddings                                       │
│  ✅ Persists across restarts                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TEMPORARY STORAGE (Lost on Restart)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  💾 response_storage: Dict[str, Dict]                       │
│     {                                                        │
│       "CAS001": { responseBody: "...", ... },               │
│       "CAS002": { responseBody: "...", ... },               │
│       ...                                                    │
│     }                                                        │
│                                                              │
│  ⚠️  Generated responses                                    │
│  ⚠️  Lost on application restart                            │
│  ⚠️  Consider using database for production                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## API Request/Response Examples

### **1. Upload Document with Category**
```http
POST /upload
Content-Type: multipart/form-data

file: pmay_faq.pdf
category: "PMAY-U 2.0"
sub_category: "Eligibility"

───────────────────────────────────────
✅ 200 OK
{
  "message": "FAQ PDF processed successfully",
  "filename": "pmay_faq.pdf",
  "chunks": 25,
  "sections": 1,
  "file_size_mb": 0.15,
  "category": "PMAY-U 2.0",
  "sub_category": "Eligibility"
}
```

### **2. Process CQR (Async - Returns Immediately)**
```http
POST /cqr
Content-Type: application/json

{
  "caseNumber": "TEST-001",
  "category": "PMAY-U 2.0",
  "subCategory": "Eligibility",
  "custName": "John Doe",
  "fileNumber": "FILE001",
  "requestBody": "What is the eligibility criteria?"
}

───────────────────────────────────────
✅ 202 Accepted (IMMEDIATE RESPONSE)
{
  "message": "Request accepted and processing in background",
  "caseNumber": "TEST-001",
  "status": "InProgress",
  "requestTimestamp": "2025-10-20T10:30:00"
}

⚠️  NOTE: Processing happens in background!
    Use /response to check status.
```

### **3. Check Status / Get Response**
```http
POST /response
Content-Type: application/json

{ "caseNumber": "TEST-001" }

───────────────────────────────────────
CASE A: Still Processing
✅ 202 Accepted
{
  "caseNumber": "TEST-001",
  "status": "InProgress",
  "message": "Request is still being processed. Please check again later."
}

───────────────────────────────────────
CASE B: Completed
✅ 200 OK
{
  "caseNumber": "TEST-001",
  "responseBody": "Families belonging to EWS/LIG/MIG segments...",
  "category": "PMAY-U 2.0",
  "subCategory": "Eligibility",
  "custName": "John Doe",
  "fileNumber": "FILE001"
}

───────────────────────────────────────
CASE C: Failed
❌ 500 Internal Server Error
{
  "detail": "Error message here"
}
```

### **4. List All Cases**
```http
GET /responses/list

───────────────────────────────────────
✅ 200 OK
{
  "total_cases": 3,
  "case_numbers": ["CAS001234", "CAS001235", "CAS001236"]
}
```

### **5. Delete Response**
```http
DELETE /response/CAS001234

───────────────────────────────────────
✅ 200 OK
{
  "message": "Response deleted successfully",
  "caseNumber": "CAS001234"
}
```

---

## Technology Stack

```
┌─────────────────────────────────────────────┐
│  Backend Framework                           │
│  • FastAPI (Python web framework)           │
│  • Pydantic (data validation)               │
│  • Uvicorn (ASGI server)                    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Vector Database                             │
│  • ChromaDB (open-source)                   │
│  • SQLite-based storage                     │
│  • Efficient vector search                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  LLM & Embeddings                            │
│  • Ollama (local LLM runtime)               │
│  • Qwen2.5 (response generation)            │
│  • nomic-embed-text (embeddings)            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  PDF Processing                              │
│  • pdfplumber (MIT License)                 │
│  • LangChain (text splitting)               │
│  • RecursiveCharacterTextSplitter           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Storage                                     │
│  • In-Memory: Python Dict (temp storage)    │
│  • Persistent: ChromaDB (FAQ knowledge base)│
└─────────────────────────────────────────────┘
```

---

## Deployment Considerations

### **For Production:**

1. **Replace In-Memory Storage:**
   ```python
   # Current (loses data on restart):
   response_storage: Dict[str, Dict] = {}
   
   # Production alternatives:
   # - PostgreSQL (full database)
   # - MongoDB (document store)
   # - Redis (fast in-memory DB with persistence)
   # - SQLite (simple file-based DB)
   ```

2. **Add Authentication:**
   - API key authentication
   - JWT tokens
   - OAuth2

3. **Add Rate Limiting:**
   - Prevent abuse
   - Throttle requests

4. **Add Monitoring:**
   - Logging
   - Metrics
   - Error tracking

5. **Scale ChromaDB:**
   - Consider hosted vector DB (Pinecone, Weaviate)
   - Or self-hosted Milvus/Qdrant

---

## Quick Start Commands

```bash
# 1. Start server
python main.py

# 2. Upload FAQ
curl -X POST "http://localhost:8000/upload" \
  -F "file=@faq.pdf"

# 3. Process query
curl -X POST "http://localhost:8000/cqr" \
  -H "Content-Type: application/json" \
  -d '{"caseNumber": "CAS001", ...}'

# 4. Get response
curl -X GET "http://localhost:8000/response/CAS001"

# 5. List all cases
curl -X GET "http://localhost:8000/responses/list"
```
