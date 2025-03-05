import os
import json
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from PIL import Image
import base64
from io import BytesIO
import fitz  # PyMuPDF
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# PDF path
PDF_PATH = "B-12-Manual.pdf"

# Set up ChromaDB
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def load_collection(collection_name="pdf_qa_collection"):
    client = get_chroma_client()
    embedding_func = get_embedding_function()
    
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        return collection
    except ValueError:
        st.error(f"Collection '{collection_name}' not found. Please make sure to build the index first.")
        st.stop()

def query_collection(collection, query, n_results=3):
    """Query the collection and return the most relevant chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    processed_results = []
    
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        score = results["distances"][0][i] if "distances" in results else 0
        
        # Find the corresponding chunk file
        chunk_file = f"chunks/chunk_{chunk_id}.json"
        associated_images = []
        
        if os.path.exists(chunk_file):
            with open(chunk_file, "r") as f:
                chunk_data = json.load(f)
                
                # Get associated images
                for img in chunk_data.get("associated_images", []):
                    img_path = img.get("image_path", "")
                    if os.path.exists(img_path):
                        associated_images.append({
                            "path": img_path,
                            "base64": image_to_base64(img_path)
                        })
        
        processed_results.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "score": score,
            "images": associated_images,
            "page_num": metadata.get("page_num", 0)
        })
    
    # Sort results by score (lower distance means higher relevance)
    # In vector databases, lower distance/score means higher similarity
    processed_results = sorted(processed_results, key=lambda x: x.get("score", float('inf')))
    
    # Log the scores for debugging
    print("Sorted chunks by score:")
    for i, result in enumerate(processed_results):
        print(f"Chunk {i+1}: ID={result['chunk_id']}, Score={result['score']}, Page={result['page_num']}")
    
    return processed_results

def image_to_base64(image_path):
    """Convert an image to base64 for embedding in HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def generate_answer(query, context_chunks):
    """Generate an answer using Gemini API based on the retrieved chunks."""
    # Prepare context from chunks
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    
    # Create prompt for Gemini
    prompt = f"""
    You are an assistant that answers questions based on the provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    Please provide a detailed and accurate answer based only on the information in the context.
    If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer this question."
    """
    
    # Generate response
    response = model.generate_content(prompt)
    return response.text

def highlight_pdf_text(pdf_path, chunks):
    """Create a PDF with only the page containing the top scoring chunk, with the text highlighted."""
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        return None, "PDF file not found."
    
    # Create a temporary file for the highlighted PDF
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "highlighted.pdf")
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # The chunks should already be sorted by score from query_collection
        # But let's ensure they are sorted correctly (lower score is better)
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", float('inf')))
        
        if not sorted_chunks:
            return None, "No chunks found."
        
        # Get the top scoring chunk
        top_chunk = sorted_chunks[0]
        page_num = top_chunk.get("page_num", 0)
        text = top_chunk.get("text", "")
        
        # Log the top chunk for debugging
        print(f"Top chunk selected for highlighting:")
        print(f"  ID: {top_chunk.get('chunk_id')}")
        print(f"  Score: {top_chunk.get('score')}")
        print(f"  Page: {page_num}")
        print(f"  Text preview: {text[:100]}...")
        
        # Make sure page_num is valid
        if 0 <= page_num < len(doc):
            # Create a new PDF with just this page
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Get the page in the new document (it's now page 0)
            page = new_doc[0]
            
            # Improved text search strategy
            highlighted = False
            
            # Try to find the exact text first
            if len(text) <= 500:  # For reasonably sized chunks, try exact match first
                text_instances = page.search_for(text)
                if text_instances:
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors({"stroke": (1, 0.8, 0)})  # Yellow highlight
                        highlight.update()
                    highlighted = True
                    print(f"  Highlighted using exact text match")
            
            # If exact match fails or text is too long, try with sentences
            if not highlighted:
                # Split text into sentences and try to find each sentence
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                
                for sentence in sentences:
                    if len(sentence) > 10:  # Only search for substantial sentences
                        text_instances = page.search_for(sentence)
                        for inst in text_instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors({"stroke": (1, 0.8, 0)})  # Yellow highlight
                            highlight.update()
                            highlighted = True
                
                if highlighted:
                    print(f"  Highlighted using sentence matching")
            
            # If still not highlighted, fall back to first 100 chars as a last resort
            if not highlighted:
                text_instances = page.search_for(text[:100])
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors({"stroke": (1, 0.8, 0)})  # Yellow highlight
                    highlight.update()
                print(f"  Highlighted using first 100 chars fallback")
            
            # Save the highlighted PDF
            new_doc.save(output_path)
            new_doc.close()
        else:
            return None, f"Invalid page number: {page_num}"
        
        doc.close()
        
        return output_path, None
    except Exception as e:
        return None, f"Error highlighting PDF: {str(e)}"

# Streamlit UI
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.subheader("Ask questions about the technical manual")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for highlighted PDF
if "highlighted_pdf" not in st.session_state:
    st.session_state.highlighted_pdf = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display images if any
        if "images" in message and message["images"]:
            cols = st.columns(min(3, len(message["images"])))
            for i, img_data in enumerate(message["images"]):
                with cols[i % 3]:
                    st.image(img_data["path"], caption=f"Image {i+1}")

# Chat input
query = st.chat_input("Ask a question about the technical manual...")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Load collection
                collection = load_collection()
                
                # Query collection
                results = query_collection(collection, query, n_results=3)
                
                # Generate answer
                answer = generate_answer(query, results)
                
                # Display answer
                st.markdown(answer)
                
                # Display images if any
                all_images = []
                for result in results:
                    all_images.extend(result["images"])
                
                if all_images:
                    st.subheader("Related Images:")
                    cols = st.columns(min(3, len(all_images)))
                    for i, img_data in enumerate(all_images):
                        with cols[i % 3]:
                            st.image(img_data["path"], caption=f"Image {i+1}")
                
                # Create highlighted PDF
                highlighted_pdf_path, error = highlight_pdf_text(PDF_PATH, results)
                if highlighted_pdf_path:
                    st.session_state.highlighted_pdf = highlighted_pdf_path
                    
                    # Store top chunk info for display
                    if results:
                        top_chunk = results[0]  # First result is the highest scored
                        st.session_state.top_chunk_info = {
                            'id': top_chunk.get('chunk_id', 'Unknown'),
                            'score': top_chunk.get('score', 0),
                            'page': top_chunk.get('page_num', 0)
                        }
                elif error:
                    st.error(error)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "images": all_images
                })

with col2:
    st.subheader("Source Document")
    
    # Display the highlighted PDF if available
    if st.session_state.highlighted_pdf and os.path.exists(st.session_state.highlighted_pdf):
        with open(st.session_state.highlighted_pdf, "rb") as f:
            pdf_bytes = f.read()
        
        st.download_button(
            label="Download Highlighted Page",
            data=pdf_bytes,
            file_name="highlighted_page.pdf",
            mime="application/pdf"
        )
        
        # Get the top chunk info if available
        if 'top_chunk_info' not in st.session_state:
            st.session_state.top_chunk_info = None
            
        # Display info about what's being shown
        st.info("Showing the page containing the most relevant information. The exact text chunk used for the answer is highlighted in yellow.")
        
        # If we have top chunk info, display it
        if st.session_state.top_chunk_info:
            with st.expander("Top Chunk Details"):
                st.write(f"Chunk ID: {st.session_state.top_chunk_info['id']}")
                st.write(f"Relevance Score: {st.session_state.top_chunk_info['score']:.6f}")
                st.write(f"Page Number: {st.session_state.top_chunk_info['page']}")
        
        # Display PDF using iframe
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        # Display original PDF if no highlighted version
        if os.path.exists(PDF_PATH):
            with open(PDF_PATH, "rb") as f:
                pdf_bytes = f.read()
            
            # Display PDF using iframe
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.info("PDF document not found.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about the technical manual.
    
    It retrieves relevant chunks of text from the document and uses Gemini to generate answers.
    
    If there are images associated with the retrieved text, they will be displayed alongside the answer.
    
    The source document view shows only the page containing the most relevant information, with the exact text chunk highlighted in yellow.
    """)
    
    st.header("Build Index")
    if st.button("Rebuild Index"):
        st.info("This will rebuild the vector index from the PDF. This may take a few minutes.")
        
        # Run the pdf_text_image_association.py script
        with st.spinner("Building index..."):
            os.system("python pdf_text_image_association.py")
            os.system("python build_index.py")
            st.success("Index built successfully!") 