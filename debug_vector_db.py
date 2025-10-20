"""
Debug script to check vector database and test retrieval
"""
from rag_service import RAGService
import json

print("=" * 80)
print("VECTOR DATABASE DEBUG TOOL")
print("=" * 80)

# Initialize RAG service
print("\n1. Initializing RAG Service...")
rag_service = RAGService()

# Get collection info
print("\n2. Checking Vector Database Collection...")
try:
    collection = rag_service.vectorstore._collection
    count = collection.count()
    print(f"   ✓ Total documents in vector DB: {count}")
    
    # Get a sample of documents
    if count > 0:
        print("\n3. Sample Documents in Vector DB:")
        results = collection.get(limit=5, include=['documents', 'metadatas'])
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\n   Document {i+1}:")
            print(f"   - Source: {meta.get('source', 'Unknown')}")
            print(f"   - Page: {meta.get('page', 'Unknown')}")
            print(f"   - Category: {meta.get('category', 'Not set')}")
            print(f"   - Sub-Category: {meta.get('sub_category', 'Not set')}")
            print(f"   - Content preview: {doc[:200]}...")
    else:
        print("   ⚠️  No documents found in vector database!")
        print("   Please upload some documents first using the /upload endpoint")
        exit(0)
        
except Exception as e:
    print(f"   ❌ Error accessing collection: {e}")
    exit(1)

# Test query
print("\n" + "=" * 80)
print("4. Testing Query Retrieval")
print("=" * 80)

test_queries = [
    "What is the eligibility criteria for seeking benefit under the scheme?",
    "eligibility criteria PMAY",
    "EWS LIG MIG segments urban areas pucca house",
]

for query in test_queries:
    print(f"\n📝 Query: {query}")
    print("-" * 80)
    
    try:
        # Test with different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            print(f"\n   Threshold: {threshold}")
            result = rag_service.query(query, k=5, score_threshold=threshold)
            
            chunks = result.get('sources', [])
            print(f"   - Chunks retrieved: {len(chunks)}")
            
            if chunks:
                print(f"   - Top chunk similarity: {chunks[0].get('score', 'N/A')}")
                print(f"   - Source: {chunks[0].get('source', 'Unknown')}")
                print(f"   - Content preview: {chunks[0].get('content', '')[:150]}...")
            else:
                print(f"   ⚠️  No chunks retrieved with threshold {threshold}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Direct similarity search without threshold
print("\n" + "=" * 80)
print("5. Direct Similarity Search (No Threshold)")
print("=" * 80)

query = "What is the eligibility criteria for seeking benefit under the scheme?"
print(f"\n📝 Query: {query}")

try:
    # Use the vectorstore directly for similarity search with scores
    docs_with_scores = rag_service.vectorstore.similarity_search_with_score(query, k=10)
    
    print(f"\n   Found {len(docs_with_scores)} results:")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n   Result {i}:")
        print(f"   - Similarity Score: {score:.4f}")
        print(f"   - Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"   - Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"   - Category: {doc.metadata.get('category', 'Not set')}")
        print(f"   - Chunk: {doc.metadata.get('chunk', 'Unknown')}")
        print(f"   - Content: {doc.page_content[:200]}...")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
