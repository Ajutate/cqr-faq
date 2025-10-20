# CQR-FAQ API Documentation

## Overview
This API provides a Customer Query Response (CQR) system for HDFC Home Loan FAQs. It uses RAG (Retrieval-Augmented Generation) with an open-source vector database (ChromaDB) to answer customer queries based on a knowledge base of FAQ documents.

## Architecture

### Components:
1. **Vector Database**: ChromaDB (open-source) for FAQ knowledge base storage
2. **LLM**: Ollama Qwen2.5 model for generating responses
3. **In-Memory Storage**: Python Dictionary for storing generated responses by case number
4. **RAG Service**: Retrieves relevant FAQ context before generating responses

---

## API Endpoints

### 1. **Upload FAQ PDF** 
**POST** `/upload`

Upload FAQ PDF documents to build the knowledge base.

**Request:**
- Content-Type: `multipart/form-data`
- Body: PDF file

**Response:**
```json
{
  "message": "FAQ PDF processed successfully and added to knowledge base",
  "filename": "home_loan_faq.pdf",
  "chunks": 150,
  "sections": 25,
  "file_size_mb": 2.5
}
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@home_loan_faq.pdf"
```

---

### 2. **Process CQR Request (Void Endpoint)**
**POST** `/cqr`

Process a customer query and generate a response using RAG. The response is stored in memory and NOT returned to the caller.

**Request Body:**
```json
{
  "caseNumber": "CAS001234",
  "category": "Interest Rate",
  "subCategory": "ROI Inquiry",
  "custName": "John Doe",
  "fileNumber": "FILE567890",
  "requestBody": "What is the current interest rate for home loans?"
}
```

**Response:**
```json
{
  "message": "Request processed successfully",
  "caseNumber": "CAS001234",
  "status": "stored"
}
```

**Flow:**
1. Receives customer query with category information
2. Queries FAQ knowledge base using RAG (retrieves relevant FAQ context)
3. Generates professional email response using LLM with FAQ context
4. Stores response in in-memory dictionary
5. Returns acknowledgment only (not the actual response)

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/cqr" \
  -H "Content-Type: application/json" \
  -d '{
    "caseNumber": "CAS001234",
    "category": "Interest Rate",
    "subCategory": "ROI Inquiry",
    "custName": "John Doe",
    "fileNumber": "FILE567890",
    "requestBody": "What is the current interest rate for home loans?"
  }'
```

---

### 3. **Get Stored Response**
**GET** `/response/{caseNumber}`

Retrieve the stored response for a specific case number.

**Path Parameter:**
- `caseNumber`: Case number to retrieve

**Response:**
```json
{
  "caseNumber": "CAS001234",
  "responseBody": "Dear John Doe,\n\nRegarding your case CAS001234 and file FILE567890...\n\nBest regards,\nHDFC Home Loan Support Team",
  "category": "Interest Rate",
  "subCategory": "ROI Inquiry",
  "custName": "John Doe",
  "fileNumber": "FILE567890"
}
```

**Error Response (404):**
```json
{
  "detail": "No response found for case number: CAS001234"
}
```

**Example (cURL):**
```bash
curl -X GET "http://localhost:8000/response/CAS001234"
```

---

### 4. **List All Stored Responses**
**GET** `/responses/list`

List all case numbers that have stored responses in memory.

**Response:**
```json
{
  "total_cases": 5,
  "case_numbers": [
    "CAS001234",
    "CAS001235",
    "CAS001236",
    "CAS001237",
    "CAS001238"
  ]
}
```

**Example (cURL):**
```bash
curl -X GET "http://localhost:8000/responses/list"
```

---

### 5. **Delete Stored Response**
**DELETE** `/response/{caseNumber}`

Delete a stored response from in-memory storage.

**Path Parameter:**
- `caseNumber`: Case number to delete

**Response:**
```json
{
  "message": "Response deleted successfully",
  "caseNumber": "CAS001234"
}
```

**Error Response (404):**
```json
{
  "detail": "No response found for case number: CAS001234"
}
```

**Example (cURL):**
```bash
curl -X DELETE "http://localhost:8000/response/CAS001234"
```

---

## Workflow Example

### Complete Workflow:

1. **Upload FAQ PDF to Knowledge Base:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@hdfc_home_loan_faq.pdf"
```

2. **Process Customer Query (stores response):**
```bash
curl -X POST "http://localhost:8000/cqr" \
  -H "Content-Type: application/json" \
  -d '{
    "caseNumber": "CAS001234",
    "category": "Loan Eligibility",
    "subCategory": "Income Criteria",
    "custName": "Jane Smith",
    "fileNumber": "FILE789012",
    "requestBody": "What is the minimum income required for a home loan?"
  }'
```

Response:
```json
{
  "message": "Request processed successfully",
  "caseNumber": "CAS001234",
  "status": "stored"
}
```

3. **Retrieve Stored Response:**
```bash
curl -X GET "http://localhost:8000/response/CAS001234"
```

Response:
```json
{
  "caseNumber": "CAS001234",
  "responseBody": "Dear Jane Smith,\n\nRegarding your case CAS001234 and file FILE789012...\n\nBased on our current policies...\n\nBest regards,\nHDFC Home Loan Support Team",
  "category": "Loan Eligibility",
  "subCategory": "Income Criteria",
  "custName": "Jane Smith",
  "fileNumber": "FILE789012"
}
```

---

## Technical Details

### In-Memory Storage
- Uses Python `Dict[str, Dict]` for storing responses
- Key: `caseNumber`
- Value: Complete response data including metadata
- **Note**: Data is lost on application restart (in-memory only)

### Vector Database (ChromaDB)
- Open-source vector database
- Stores FAQ document embeddings
- Location: `./chroma_db/`
- Uses Ollama embeddings (nomic-embed-text model)

### RAG Process:
1. Customer query is combined with category/sub-category
2. Query is embedded and searched in vector database
3. Top 5 most relevant FAQ chunks are retrieved
4. Retrieved context is passed to LLM along with customer query
5. LLM generates contextual response based on FAQ knowledge

---

## Environment Variables

Create a `.env` file with:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with required models
ollama pull qwen2.5:latest
ollama pull nomic-embed-text

# Run the application
python main.py
```

The API will be available at: `http://localhost:8000`

API Documentation (Swagger): `http://localhost:8000/docs`

---

## Key Features

✅ **RAG-based FAQ Retrieval**: Uses vector database to retrieve relevant FAQ context  
✅ **Category-aware Responses**: Queries are filtered by category and sub-category  
✅ **In-Memory Storage**: Responses stored in dictionary for async retrieval  
✅ **Open-Source Stack**: ChromaDB + Ollama (no proprietary APIs needed)  
✅ **Professional Email Format**: Generates customer-ready email responses  
✅ **Void POST Endpoint**: `/cqr` doesn't return response directly  
✅ **Separate GET Endpoint**: Retrieve responses by case number  

---

## Important Notes

⚠️ **In-Memory Storage Limitation**: 
- All stored responses are lost when the application restarts
- For persistent storage, consider using a database (PostgreSQL, MongoDB, etc.)

⚠️ **Vector Database Persistence**:
- FAQ knowledge base persists across restarts (stored in `./chroma_db/`)
- Upload FAQs once, they remain available until explicitly cleared

⚠️ **Concurrent Processing**:
- Multiple CQR requests can be processed simultaneously
- Each case number should be unique to avoid overwrites
