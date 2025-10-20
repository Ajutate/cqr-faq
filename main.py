from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from typing import Dict, Optional
from datetime import datetime

from langchain_ollama import OllamaLLM
from rag_service import RAGService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="CQR API Application")

# In-memory storage for responses (Dictionary)
response_storage: Dict[str, Dict] = {}

# In-memory storage for processing status (caseNumber -> status)
processing_status: Dict[str, str] = {}  # Values: "InProgress", "Completed", "Failed"

# Initialize RAG service for FAQ knowledge base
rag_service = RAGService()

# Pydantic models for CQR endpoint
class CQRRequest(BaseModel):
    caseNumber: str
    category: str
    subCategory: str
    custName: str
    fileNumber: str
    requestBody: str


class CQRResponse(BaseModel):
    caseNumber: str
    responseBody: str


# Initialize Ollama LLM for CQR endpoint
cqr_llm = OllamaLLM(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model="qwen3:latest",  # Using llama3.1:8b model
    temperature=0.0,  # Set to 0 for maximum determinism and accuracy - NO creativity
    num_predict=2048,  # Allow longer responses (default is usually 128-512)
)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    category: str = Form(None),
    sub_category: str = Form(None)
):
    """
    Upload and process document file to knowledge base with optional categorization
    
    Form Parameters:
    - file: Document file - PDF, DOC, DOCX, or TXT (required)
    - category: Optional category (e.g., "Home Loan", "Personal Loan", "Credit Card")
    - sub_category: Optional sub-category (e.g., "Eligibility", "Documentation", "Processing")
    """
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={"error": f"Only {', '.join(allowed_extensions)} files are allowed"}
            )
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        print(f"📁 Uploading document: {file.filename} ({file_size_mb:.2f} MB)")
        if category:
            print(f"   Category: {category}")
        if sub_category:
            print(f"   Sub-Category: {sub_category}")
        
        # Save uploaded file
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        print(f"💾 File saved, processing into knowledge base...")
        
        # Process document and store in vector DB with optional category metadata
        result = rag_service.process_document(file_path, category=category, sub_category=sub_category)
        
        response_data = {
            "message": f"Document processed successfully and added to knowledge base",
            "filename": file.filename,
            "file_type": file_extension,
            "chunks": result["chunks"],
            "sections": result.get("sections", 0),
            "file_size_mb": round(file_size_mb, 2)
        }
        
        # Include category info in response if provided
        if category:
            response_data["category"] = category
        if sub_category:
            response_data["sub_category"] = sub_category
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error processing document: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing document: {str(e)}"}
        )


def process_cqr_background(request: CQRRequest):
    """
    Background task to process CQR request
    """
    try:
        # Start timing
        start_time = datetime.now()
        
        print(f"📧 Processing CQR request for Case: {request.caseNumber}")
        print(f"   Category: {request.category} | Sub-Category: {request.subCategory}")
        
        # Query the FAQ knowledge base using RAG based on category and question
        # Using similarity threshold of 0.5 (adjust higher for stricter matching)
        faq_query = f"Category: {request.category}, Sub-Category: {request.subCategory}. Question: {request.requestBody}"
        
        # Get relevant FAQ information from vector database
        # score_threshold: 0.3 = cast wider net to ensure correct chunk is retrieved
        # k: 8 = retrieve more chunks to increase chance of getting the right answer
        # Lower threshold + more chunks = better recall (ensure we find the answer)
        rag_result = rag_service.query(faq_query, k=8, score_threshold=0.3)
        
        # Build context from raw source chunks (not from rag_service's pre-generated answer)
        # This gives our LLM direct access to the FAQ content without intermediate interpretation
        raw_chunks = rag_result.get('sources', [])
        if raw_chunks:
            faq_context = "\n\n".join([chunk['content'] for chunk in raw_chunks])
        else:
            faq_context = "No relevant information found in the knowledge base."
        
        print(f"📚 Retrieved FAQ context from knowledge base")
        print(f"   Sources: {rag_result.get('source_files', [])}")
        print(f"   Chunks retrieved: {len(raw_chunks)}")
        
        # Print first 200 chars of retrieved context for debugging
        print(f"   Context preview: {faq_context[:200]}...")
        
        # Build enhanced prompt with FAQ context
        prompt = f"""You are a STRICT TEXT EXTRACTOR. Your ONLY job is to find ONE specific FAQ answer and copy it EXACTLY.

CUSTOMER'S QUESTION:
{request.requestBody}

REFERENCE TEXT (Multiple FAQ Q&A pairs):
{faq_context}

EXTRACTION PROCESS:
1. Search the reference for a question that matches the customer's question
2. When you find the matching question, identify where its answer ENDS (before the next question starts)
3. Copy ONLY that specific answer - NOTHING MORE, NOTHING LESS
4. Use the EXACT words from the reference - DO NOT add, remove, or change ANY words
5. DO NOT include the question itself in your output
6. DO NOT include information from other FAQ entries
7. DO NOT add explanations, clarifications, or additional details
8. DO NOT merge or combine answers from multiple FAQs

CRITICAL RULES:
- If you see "EWS/LIG/MIG" in the answer, copy it exactly as written
- If you see income limits in a DIFFERENT FAQ, DO NOT add them
- STOP at the end of the matched answer - do not continue into the next question
- Every word you output must exist in the matched answer section

WHAT TO OUTPUT:
Just the pure answer text - nothing before it, nothing after it.

Extract the answer now:"""

        # Get response from Ollama LLM model
        response_text = cqr_llm.invoke(prompt)
        
        # Minimal cleanup - preserve FAQ formatting
        response_text = response_text.strip()
        
        # Remove "Answer:" prefix if LLM added it despite instructions
        if response_text.startswith("Answer:"):
            response_text = response_text[7:].strip()
        
        # Check if answer was found
        if not response_text or "could not find" in response_text.lower() or "sorry" in response_text.lower():
            final_answer = "We are sorry, but we could not find relevant information to answer your question at this time."
        else:
            final_answer = response_text
        
        # Format the response as a clean email with greeting and signature
        email_body = f""" {final_answer}"""

        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Store response in in-memory dictionary
        current_timestamp = end_time.isoformat()
        response_storage[request.caseNumber] = {
            
            "caseNumber": request.caseNumber,
            "responseBody": email_body,
            "category": request.category,
            "subCategory": request.subCategory,
            "custName": request.custName,
            "fileNumber": request.fileNumber,
            "requestBody": request.requestBody,
            "requestTimestamp": start_time.isoformat(),
            "responseTimestamp": current_timestamp,
            "processingTimeSeconds": round(processing_time, 2)
        }
        
        print(f"✅ CQR response generated and stored for Case: {request.caseNumber}")
        print(f"⏰ Timestamp: {current_timestamp}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"💾 Response stored in memory (total cases: {len(response_storage)})")
        print(f"💾 Response Body: {email_body}")
        
        # Mark as completed
        processing_status[request.caseNumber] = "Completed"
        print(f"✅ Status updated to: Completed")
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error processing CQR request: {error_detail}")
        
        # Mark as failed
        processing_status[request.caseNumber] = "Failed"
        response_storage[request.caseNumber] = {
            "caseNumber": request.caseNumber,
            "responseBody": f"Error: {str(e)}",
            "status": "Failed",
            "error": str(e)
        }


@app.post("/cqr")
async def process_cqr(request: CQRRequest, background_tasks: BackgroundTasks):
    """
    CQR endpoint - Process home loan FAQ questions using RAG asynchronously
    Returns immediately with 202 Accepted status
    Check /response endpoint with caseNumber to get the result
    """
    try:
        # Mark as in progress immediately
        processing_status[request.caseNumber] = "InProgress"
        
        # Add background task
        background_tasks.add_task(process_cqr_background, request)
        
        print(f"📧 CQR request accepted for Case: {request.caseNumber}")
        print(f"   Status: InProgress")
        print(f"   Processing in background...")
        
        # Return immediate acknowledgment
        return JSONResponse(
            status_code=202,  # 202 Accepted
            content={
                "message": "Request accepted and processing in background",
                "caseNumber": request.caseNumber,
                "status": "InProgress",
                "requestTimestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        print(f"❌ Error accepting CQR request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error accepting CQR request: {str(e)}"
        )


@app.post("/response", response_model=CQRResponse)
async def get_response(request: dict):
    """
    POST endpoint to retrieve stored response by case number (sent in request body)
    Request body: {"caseNumber": "HL-2025-0001"}
    """
    try:
        caseNumber = request.get("caseNumber")
        
        if not caseNumber:
            raise HTTPException(
                status_code=400,
                detail="caseNumber is required in request body"
            )
        
        print(f"🔍 Retrieving response for Case: {caseNumber}")
        
        # Check processing status first
        status = processing_status.get(caseNumber)
        
        if status == "InProgress":
            # Still processing - return InProgress status
            print(f"⏳ Case {caseNumber} is still in progress")
            return JSONResponse(
                status_code=202,  # 202 Accepted - still processing
                content={
                    "caseNumber": caseNumber,
                    "status": "InProgress",
                    "message": "Request is still being processed. Please check again later."
                }
            )
        
        elif status == "Failed":
            # Processing failed
            print(f"❌ Case {caseNumber} processing failed")
            stored_response = response_storage.get(caseNumber, {})
            raise HTTPException(
                status_code=500,
                detail=stored_response.get("error", "Processing failed")
            )
        
        elif status == "Completed":
            # Processing completed - return the response
            if caseNumber not in response_storage:
                raise HTTPException(
                    status_code=404,
                    detail=f"Response completed but data not found for case number: {caseNumber}"
                )
            
            stored_response = response_storage[caseNumber]
            print(f"✅ Response retrieved for Case: {caseNumber}")
            
            return CQRResponse(
                caseNumber=stored_response["caseNumber"],
                responseBody=stored_response["responseBody"]
            )
        
        else:
            # Case number not found at all
            raise HTTPException(
                status_code=404,
                detail=f"No request found for case number: {caseNumber}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error retrieving response: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving response: {str(e)}"
        )


@app.get("/responses/list")
async def list_all_responses():
    """
    GET endpoint to list all stored case numbers
    """
    try:
        case_numbers = list(response_storage.keys())
        response_storage_values = list(response_storage.values())
        print(f"📋 Listing all stored responses: {len(case_numbers)} cases")
        
        return JSONResponse(content={
            "total_cases": len(case_numbers),
            "case_numbers": case_numbers,
            "responses": response_storage_values
        })
    
    except Exception as e:
        print(f"❌ Error listing responses: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error listing responses: {str(e)}"}
        )


@app.delete("/response/{caseNumber}")
async def delete_response(caseNumber: str):
    """
    DELETE endpoint to remove a stored response by case number
    """
    try:
        if caseNumber not in response_storage:
            raise HTTPException(
                status_code=404,
                detail=f"No response found for case number: {caseNumber}"
            )
        
        del response_storage[caseNumber]
        print(f"🗑️ Deleted response for Case: {caseNumber}")
        
        return JSONResponse(content={
            "message": "Response deleted successfully",
            "caseNumber": caseNumber
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting response: {str(e)}"
        )


@app.post("/debug/query")
async def debug_query(request: dict):
    """
    Debug endpoint to test RAG queries and see retrieved chunks
    """
    try:
        query = request.get("query", "")
        k = request.get("k", 5)
        score_threshold = request.get("score_threshold", 0.45)  # Default 0.45 for balanced precision/recall
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query is required"}
            )
        
        print(f"🔍 Debug Query: {query}")
        print(f"   Similarity threshold: {score_threshold}")
        
        # Query the RAG system with threshold
        result = rag_service.query(query, k=k, score_threshold=score_threshold)
        
        # Return detailed information
        return JSONResponse(content={
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "source_files": result.get("source_files", []),
            "chunks_retrieved": len(result["sources"]),
            "score_threshold": score_threshold
        })
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error in debug query: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing query: {str(e)}"}
        )



@app.get("/debug/vector_chunks")
async def debug_vector_chunks(filename: str = None, include_embeddings: bool = False, limit: int = 50):
    """
    Return chunk-level information from the ChromaDB vector store.

    Query params:
    - filename: optional source filename to filter chunks (e.g., 'About Pradhan Mantri Awas Yojana - Urban 2.0_2.pdf')
    - include_embeddings: if true, include the first 32 dimensions of each embedding
    - limit: max number of chunks to return (default 50)
    """
    try:
        # Access the underlying collection via RAG service
        collection = rag_service.vectorstore._collection

        # Build include list - ChromaDB returns ids by default, don't include it in the list
        include = ['documents', 'metadatas']
        if include_embeddings:
            include.append('embeddings')

        # Query collection
        if filename:
            # Filter by source filename - ChromaDB filter format: {"source": value}
            results = collection.get(where={"source": filename}, include=include, limit=limit)
        else:
            results = collection.get(include=include, limit=limit)

        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        embeddings = results.get('embeddings', None)

        items = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            item = {
                'id': ids[i] if i < len(ids) else None,
                'source': meta.get('source', None),
                'page': meta.get('page', None),
                'chunk': meta.get('chunk', None),
                'category': meta.get('category', None),
                'sub_category': meta.get('sub_category', None),
                'method': meta.get('method', None),
                'content_preview': doc[:1000] if doc else None
            }

            if include_embeddings and embeddings is not None and len(embeddings) > 0:
                emb = embeddings[i] if i < len(embeddings) else None
                if emb is not None:
                    # include a preview of the embedding to avoid huge payloads
                    # Convert to list for JSON serialization (might be numpy array)
                    item['embedding_preview'] = list(emb[:32])
                    item['embedding_dim'] = len(emb)
                else:
                    item['embedding_preview'] = None
                    item['embedding_dim'] = 0

            items.append(item)

        return JSONResponse(content={
            'total_returned': len(items),
            'items': items
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})



if __name__ == "__main__":
    # reload=False to avoid Windows multiprocessing issues with colorama
    # For development with auto-reload, run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Changed from True to avoid Windows colorama/multiprocessing errors
        timeout_keep_alive=300  # 5 minutes timeout for large PDFs
    )
