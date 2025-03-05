import os
import json
import chromadb
from chromadb.utils import embedding_functions
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_chunks(chunks_dir="chunks"):
    """Load all chunks from the chunks directory."""
    all_chunks_path = os.path.join(chunks_dir, "all_chunks.json")
    
    if not os.path.exists(all_chunks_path):
        logger.error(f"All chunks file not found: {all_chunks_path}")
        logger.info("Please run pdf_text_image_association.py first to generate chunks.")
        return None
    
    with open(all_chunks_path, "r") as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks from {all_chunks_path}")
    return chunks

def build_vector_db(chunks, collection_name="pdf_qa_collection"):
    """Build a vector database from the chunks."""
    if not chunks:
        logger.error("No chunks provided.")
        return False
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get collection
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        # Delete if exists
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except chromadb.errors.InvalidCollectionException:
        # Collection doesn't exist yet
        logger.info(f"Collection {collection_name} does not exist yet, will create new.")
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    
    # Prepare data for batch addition
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        text = chunk["text"]
        
        # Create metadata
        metadata = {
            "page_num": chunk["page_num"],
            "has_images": len(chunk.get("associated_images", [])) > 0
        }
        
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)
    
    # Add documents to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    logger.info(f"Added {len(ids)} documents to collection {collection_name}")
    return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build a vector database from extracted PDF chunks.")
    parser.add_argument("--chunks-dir", default="chunks", help="Directory containing the extracted chunks")
    parser.add_argument("--collection-name", default="pdf_qa_collection", help="Name of the vector collection")
    args = parser.parse_args()
    
    logger.info("Starting to build vector database...")
    
    # Load chunks
    chunks = load_chunks(chunks_dir=args.chunks_dir)
    if not chunks:
        return
    
    # Build vector database
    success = build_vector_db(chunks, collection_name=args.collection_name)
    
    if success:
        logger.info("Vector database built successfully!")
    else:
        logger.error("Failed to build vector database.")

if __name__ == "__main__":
    main() 