"""
Comprehensive ChromaDB inspection tool
Check if data is properly vectorized and stored
"""
import chromadb
from chromadb.config import Settings
import numpy as np
import os

print("=" * 80)
print("CHROMADB INSPECTION TOOL")
print("=" * 80)

# Connect to ChromaDB
chroma_path = "./chroma_db"
print(f"\n1. Connecting to ChromaDB at: {chroma_path}")

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(anonymized_telemetry=False)
)

# List all collections
print("\n2. Available Collections:")
collections = client.list_collections()
print(f"   Total collections: {len(collections)}")
for i, col in enumerate(collections, 1):
    print(f"   {i}. {col.name} (ID: {col.id})")

if not collections:
    print("\n❌ No collections found! Vector database is empty.")
    exit(0)

# Get the first collection (usually there's only one)
collection = collections[0]
print(f"\n3. Inspecting Collection: {collection.name}")
print(f"   Collection ID: {collection.id}")

# Get collection count
count = collection.count()
print(f"   Total documents: {count}")

if count == 0:
    print("\n❌ Collection is empty! No documents have been added.")
    exit(0)

# Get all data including embeddings
print("\n4. Retrieving All Data (including embeddings)...")
results = collection.get(
    include=['documents', 'metadatas', 'embeddings'],
    limit=count  # Get all documents
)

print(f"   Retrieved {len(results['documents'])} documents")
print(f"   Retrieved {len(results['embeddings'])} embeddings")

# Check embedding dimensions
if results['embeddings'] is not None and len(results['embeddings']) > 0:
    embedding_dim = len(results['embeddings'][0])
    print(f"   Embedding dimensions: {embedding_dim}")
    print(f"   Expected dimension: 768 (for nomic-embed-text)")
    
    if embedding_dim != 768:
        print(f"\n⚠️  WARNING: Embedding dimension mismatch!")
        print(f"   Expected: 768, Got: {embedding_dim}")
    else:
        print(f"   ✅ Embedding dimensions are correct!")
else:
    print(f"\n❌ ERROR: No embeddings found! Data is NOT vectorized!")
    exit(1)

# Detailed inspection of first 5 documents
print("\n5. Sample Documents (First 5):")
print("=" * 80)

for i in range(min(5, len(results['documents']))):
    print(f"\n--- Document {i+1} ---")
    print(f"ID: {results['ids'][i]}")
    
    # Metadata
    meta = results['metadatas'][i]
    print(f"Metadata:")
    print(f"  - Source: {meta.get('source', 'N/A')}")
    print(f"  - Page: {meta.get('page', 'N/A')}")
    print(f"  - Category: {meta.get('category', 'N/A')}")
    print(f"  - Sub-Category: {meta.get('sub_category', 'N/A')}")
    print(f"  - Chunk: {meta.get('chunk', 'N/A')}")
    print(f"  - Method: {meta.get('method', 'N/A')}")
    
    # Document content
    doc = results['documents'][i]
    print(f"Content ({len(doc)} chars):")
    print(f"  {doc[:200]}...")
    
    # Embedding info
    embedding = results['embeddings'][i]
    print(f"Embedding:")
    print(f"  - Dimension: {len(embedding)}")
    print(f"  - Min value: {min(embedding):.4f}")
    print(f"  - Max value: {max(embedding):.4f}")
    print(f"  - Mean value: {np.mean(embedding):.4f}")
    print(f"  - First 10 values: {[f'{v:.4f}' for v in embedding[:10]]}")

# Statistics
print("\n" + "=" * 80)
print("6. Collection Statistics:")
print("=" * 80)

# Count documents by source
sources = {}
categories = {}
sub_categories = {}

for meta in results['metadatas']:
    source = meta.get('source', 'Unknown')
    sources[source] = sources.get(source, 0) + 1
    
    category = meta.get('category', 'Not set')
    categories[category] = categories.get(category, 0) + 1
    
    sub_cat = meta.get('sub_category', 'Not set')
    sub_categories[sub_cat] = sub_categories.get(sub_cat, 0) + 1

print(f"\nDocuments by Source:")
for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {source}: {count} chunks")

print(f"\nDocuments by Category:")
for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {category}: {count} chunks")

print(f"\nDocuments by Sub-Category:")
for sub_cat, count in sorted(sub_categories.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {sub_cat}: {count} chunks")

# Check embedding quality
print("\n" + "=" * 80)
print("7. Embedding Quality Check:")
print("=" * 80)

embeddings_array = np.array(results['embeddings'])

print(f"\nEmbedding Statistics:")
print(f"  - Shape: {embeddings_array.shape}")
print(f"  - Global Min: {embeddings_array.min():.4f}")
print(f"  - Global Max: {embeddings_array.max():.4f}")
print(f"  - Global Mean: {embeddings_array.mean():.4f}")
print(f"  - Global Std Dev: {embeddings_array.std():.4f}")

# Check for zero embeddings (indicates vectorization failure)
zero_embeddings = np.all(embeddings_array == 0, axis=1)
if np.any(zero_embeddings):
    print(f"\n❌ WARNING: Found {zero_embeddings.sum()} embeddings that are all zeros!")
    print(f"   This indicates vectorization failure for those documents.")
else:
    print(f"\n✅ All embeddings have non-zero values (vectorization successful)")

# Check for duplicate embeddings (might indicate issues)
print(f"\nChecking for duplicate embeddings...")
unique_embeddings = np.unique(embeddings_array, axis=0)
if len(unique_embeddings) < len(embeddings_array):
    duplicates = len(embeddings_array) - len(unique_embeddings)
    print(f"⚠️  WARNING: Found {duplicates} duplicate embeddings")
    print(f"   This might indicate duplicate documents or processing issues")
else:
    print(f"✅ All embeddings are unique")

# Test similarity search
print("\n" + "=" * 80)
print("8. Testing Similarity Search:")
print("=" * 80)

test_query = "What is the eligibility criteria for seeking benefit under the scheme?"
print(f"\nTest Query: {test_query}")

# We need to generate embedding for the query
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model="nomic-embed-text"
)

print(f"\nGenerating query embedding...")
query_embedding = embeddings.embed_query(test_query)
print(f"Query embedding dimension: {len(query_embedding)}")

# Query ChromaDB
print(f"\nQuerying ChromaDB (top 5 results)...")
query_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=['documents', 'metadatas', 'distances']
)

print(f"\nTop 5 Results:")
for i, (doc, meta, distance) in enumerate(zip(
    query_results['documents'][0], 
    query_results['metadatas'][0], 
    query_results['distances'][0]
), 1):
    similarity_score = 1 / (1 + distance)
    print(f"\n{i}. Distance: {distance:.4f} | Similarity: {similarity_score:.4f}")
    print(f"   Source: {meta.get('source', 'Unknown')}")
    print(f"   Category: {meta.get('category', 'N/A')}")
    print(f"   Content: {doc[:150]}...")

print("\n" + "=" * 80)
print("✅ INSPECTION COMPLETE")
print("=" * 80)

# Summary
print("\nSUMMARY:")
print(f"✓ Total collections: {len(collections)}")
print(f"✓ Total documents: {count}")
print(f"✓ Embedding dimension: {embedding_dim}")
print(f"✓ Unique sources: {len(sources)}")
print(f"✓ Unique categories: {len(categories)}")

if embedding_dim == 768 and not np.any(zero_embeddings):
    print(f"\n✅ Vector database is properly configured and vectorized!")
else:
    print(f"\n⚠️  Issues detected - see warnings above")
