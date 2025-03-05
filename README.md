# Minimal RAG Application

A minimalist Retrieval Augmented Generation (RAG) application that processes PDF documents, extracts text and images, and enables semantic search and question answering based on the document content.

## Features

- PDF document processing and text extraction
- Image extraction and analysis from PDFs
- Text-image association for better context understanding
- Vector database integration for semantic search
- Simple web interface for querying document content

## Requirements

All dependencies are listed in `requirements.txt`. Key dependencies include:
- Python 3.9+
- FastAPI
- ChromaDB
- PyTorch
- Sentence-Transformers
- PyPDF2
- Pillow

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/deeth-coder/super-fishstick.git
   cd super-fishstick
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your Document

Place your PDF document in the project directory or in a designated folder.

### 2. Build the Vector Index

Process your PDF document and build the vector index:

```bash
python build_index.py path/to/your/document.pdf
```

This will:
- Extract text from the PDF
- Extract and process images
- Associate text with nearby images
- Create vector embeddings
- Store everything in the ChromaDB database

### 3. Run the Application

Start the web server:

```bash
python app.py
```

The application will be available at http://localhost:8000

### 4. Query Your Document

Use the web interface at http://localhost:8000 to:
- Ask questions about the content
- Search for specific information
- View associated images

## Advanced Image-Text Association

If you need more fine-grained control over how text and images are associated, you can use the specialized script:

```bash
python pdf_text_image_association.py path/to/your/document.pdf
```

## Notes

- The vector database is stored in the `chroma_db` directory
- Extracted images are saved in the `extracted_images` directory
- For large documents, the processing may take a few minutes 