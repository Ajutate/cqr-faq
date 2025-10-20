from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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


@app.post("/cqr")
async def process_cqr(request: CQRRequest):
    """
    CQR endpoint - Process home loan FAQ questions using RAG and store response in memory
    Does not return response to caller (void endpoint)
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
        # score_threshold: 0.5 = balanced precision (fewer but more relevant chunks)
        # k: 5 = retrieve fewer chunks to reduce noise and confusion
        # Higher threshold + fewer chunks = cleaner, more focused context for LLM
        rag_result = rag_service.query(faq_query, k=5, score_threshold=0.5)
        
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
        prompt = f"""You are a STRICT TEXT EXTRACTOR. Your job is to find and copy text EXACTLY as written, with ZERO modifications.

CUSTOMER'S QUESTION:
{request.requestBody}

REFERENCE TEXT (Contains multiple FAQ entries):
{faq_context}

STRICT EXTRACTION RULES - NO EXCEPTIONS:
1. Find the FAQ answer that matches the customer's question
2. Copy it WORD-FOR-WORD, EXACTLY as written in the reference
3. DO NOT combine text from different parts of the reference
4. DO NOT add any words, phrases, or explanations that are not in the matched answer
5. DO NOT paraphrase or rewrite ANY part of the answer
6. DO NOT merge information from multiple FAQ entries
7. If the answer mentions other verticals/options, copy EXACTLY what is written
8. STOP copying when the matched answer ends - do not continue into the next FAQ
9. Include ALL sentences, conditions, and details from the matched answer
10. DO NOT add connecting phrases or extra context
11. Do NOT include the question itself
12. Include Dear {request.custName} at the beginning of the response, if available. If it is null then omit

CRITICAL: If you add even ONE WORD that is not in the exact matched answer section, you have FAILED.

EXAMPLE OF FAILURE:
Reference says: "beneficiaries can avail benefits under AHP and ISS verticals"
You write: "beneficiaries can avail benefits under AHP to purchase/construct a house" ❌ WRONG

EXAMPLE OF SUCCESS:
Reference says: "beneficiaries can avail benefits under AHP and ISS verticals"
You write: "beneficiaries can avail benefits under AHP and ISS verticals" ✅ CORRECT

Your task: Find and copy the complete answer EXACTLY as written, with ZERO additions or modifications:"""

        # Get response from Ollama LLM model
        response_text = cqr_llm.invoke(prompt)
        
        # Minimal cleanup - preserve FAQ formatting
        response_text = response_text.strip()
        
        # Remove "Answer:" prefix if LLM added it despite instructions
        if response_text.startswith("Answer:"):
            response_text = response_text[7:].strip()
        
        # Format the response as a clean email
        email_body = f"""Dear {request.custName},

{response_text}

Thank you.

Best regards,
HDFC Home Loan Support Team"""

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
        # Return acknowledgment without the actual response
        return JSONResponse(content={
            "message": "Request processed successfully",
            "caseNumber": request.caseNumber,
            "responseBody": email_body,
            "status": "stored",
            "requestTimestamp": start_time.isoformat(),
            "responseTimestamp": current_timestamp,
            "processingTimeSeconds": round(processing_time, 2)
        })
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error processing CQR request: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing CQR request: {str(e)}",
                "caseNumber": request.caseNumber
            }
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
        
        # Check if case number exists in storage
        if caseNumber not in response_storage:
            raise HTTPException(
                status_code=404,
                detail=f"No response found for case number: {caseNumber}"
            )
        
        # Retrieve stored response
        stored_response = response_storage[caseNumber]
        
        print(f"✅ Response retrieved for Case: {caseNumber}")
        
        return CQRResponse(
            caseNumber=stored_response["caseNumber"],
            responseBody=stored_response["responseBody"]
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
