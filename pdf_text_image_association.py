import os
import logging
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
from collections import defaultdict
import json  # Add import for JSON serialization
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFTextImageExtractor:
    def __init__(self, pdf_path, output_dir="extracted_images"):
        """
        Initialize the extractor with the PDF path and output directory for images.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save extracted images
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Store text chunks and their associated images
        self.chunks = []
        self.chunk_image_map = defaultdict(list)
        
        # Open the PDF document
        self.doc = fitz.open(pdf_path)
        logger.info(f"Opened PDF: {pdf_path} with {len(self.doc)} pages")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.chunk_size = 200  # Updated from 150 to 200 based on test results
        self.overlap = 100     # Updated from 30 to 100 based on test results

    def extract_images(self):
        """Extract all images from the PDF with their positions."""
        image_info = []
        
        for page_num, page in enumerate(self.doc):
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image position on the page
                rect = page.get_image_bbox(img)
                
                # Save image to file
                image_filename = f"page{page_num+1}_img{img_idx+1}.png"
                image_path = os.path.join(self.output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Store image info
                image_info.append({
                    "page_num": page_num,
                    "image_path": image_path,
                    "rect": rect,  # Position on the page
                    "xref": xref   # Reference in the PDF
                })
                
                logger.info(f"Extracted image: {image_path}, Position: {rect}")
                
        return image_info

    def extract_text_chunks(self, chunk_size=200, overlap=100):
        """
        Extract text chunks from the PDF.
        
        Args:
            chunk_size (int): Approximate size of each text chunk
            overlap (int): Overlap between consecutive chunks
            
        Returns:
            list: List of dictionaries containing chunk text and position info
        """
        chunks = []
        
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            words = text.split()
            
            # Create chunks with overlap
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if not chunk_words:
                    continue
                
                chunk_text = " ".join(chunk_words)
                
                # Get the bounding boxes of the first and last words in the chunk
                # This is a simplification - in a real implementation, you'd want to 
                # get the actual bounding boxes of all words in the chunk
                first_word_rect = None
                last_word_rect = None
                
                text_instances = page.search_for(chunk_words[0])
                if text_instances:
                    first_word_rect = text_instances[0]
                
                text_instances = page.search_for(chunk_words[-1])
                if text_instances:
                    last_word_rect = text_instances[-1]
                
                # Create a chunk with position information
                chunk = {
                    "page_num": page_num,
                    "text": chunk_text,
                    "start_rect": first_word_rect,
                    "end_rect": last_word_rect,
                    "chunk_id": len(chunks)
                }
                
                chunks.append(chunk)
                logger.info(f"Created chunk {len(chunks)}: {chunk_text[:50]}...")
        
        self.chunks = chunks
        return chunks

    def associate_chunks_with_images(self, chunks, images):
        """
        Associate text chunks with images based on proximity.
        
        Args:
            chunks (list): List of text chunks
            images (list): List of image information
            
        Returns:
            dict: Mapping of chunk IDs to lists of associated images
        """
        chunk_image_map = defaultdict(list)
        
        for chunk in chunks:
            chunk_page = chunk["page_num"]
            
            # Find images on the same page
            page_images = [img for img in images if img["page_num"] == chunk_page]
            
            for img in page_images:
                # Simple proximity check - if the chunk has position info and overlaps with image
                if chunk["start_rect"] and chunk["end_rect"] and img["rect"]:
                    # Create a bounding box for the chunk
                    chunk_rect = fitz.Rect(
                        min(chunk["start_rect"].x0, chunk["end_rect"].x0),
                        min(chunk["start_rect"].y0, chunk["end_rect"].y0),
                        max(chunk["start_rect"].x1, chunk["end_rect"].x1),
                        max(chunk["start_rect"].y1, chunk["end_rect"].y1)
                    )
                    
                    # Check if the chunk and image are close to each other
                    # This is a simple proximity check - you might want to refine this
                    img_rect = img["rect"]
                    
                    # Check if rectangles intersect or are very close
                    if (chunk_rect.intersects(img_rect) or 
                        self._are_rectangles_close(chunk_rect, img_rect, threshold=100)):
                        chunk_image_map[chunk["chunk_id"]].append(img)
                        logger.info(f"Associated chunk {chunk['chunk_id']} with image {img['image_path']}")
        
        self.chunk_image_map = chunk_image_map
        return chunk_image_map

    def _are_rectangles_close(self, rect1, rect2, threshold=100):
        """Check if two rectangles are close to each other."""
        # Calculate the center points of the rectangles
        center1 = ((rect1.x0 + rect1.x1) / 2, (rect1.y0 + rect1.y1) / 2)
        center2 = ((rect2.x0 + rect2.x1) / 2, (rect2.y0 + rect2.y1) / 2)
        
        # Calculate the Euclidean distance between the centers
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        
        return distance < threshold

    def process_pdf(self, chunk_size=200, overlap=100):
        """Process the PDF to extract text chunks and images and associate them."""
        logger.info("Starting PDF processing...")
        
        # Extract images
        images = self.extract_images()
        logger.info(f"Extracted {len(images)} images from the PDF")
        
        # Extract text chunks
        chunks = self.extract_text_chunks(chunk_size, overlap)
        logger.info(f"Extracted {len(chunks)} text chunks from the PDF")
        
        # Associate chunks with images
        chunk_image_map = self.associate_chunks_with_images(chunks, images)
        
        # Log the results
        self.log_results()
        
        # Save chunks and their associated images to disk
        self.save_chunks_to_disk()
        
        return {
            "chunks": chunks,
            "images": images,
            "chunk_image_map": chunk_image_map
        }
    
    def log_results(self):
        """Log the results of the extraction and association."""
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)
        
        for chunk_id, images in self.chunk_image_map.items():
            chunk = next((c for c in self.chunks if c["chunk_id"] == chunk_id), None)
            if chunk:
                logger.info("\n" + "-"*80)
                logger.info(f"CHUNK {chunk_id} (Page {chunk['page_num'] + 1}):")
                logger.info(f"TEXT: {chunk['text'][:200]}...")
                logger.info(f"ASSOCIATED IMAGES ({len(images)}):")
                
                for img in images:
                    logger.info(f"  - {img['image_path']} (Position: {img['rect']})")
        
        # Log chunks with no associated images
        chunks_without_images = [c for c in self.chunks if c["chunk_id"] not in self.chunk_image_map]
        logger.info("\n" + "-"*80)
        logger.info(f"CHUNKS WITHOUT ASSOCIATED IMAGES: {len(chunks_without_images)}")
        
        # Log images with no associated chunks
        all_associated_images = set()
        for images in self.chunk_image_map.values():
            for img in images:
                all_associated_images.add(img["xref"])
        
        images_without_chunks = [img for img in self.extract_images() if img["xref"] not in all_associated_images]
        logger.info(f"IMAGES WITHOUT ASSOCIATED CHUNKS: {len(images_without_chunks)}")
        logger.info("="*80)

    def simulate_retrieval(self, query_text):
        """
        Simulate retrieving a chunk based on a query and show its associated images.
        This is a simple simulation - in a real system, you'd use vector similarity.
        
        Args:
            query_text (str): The query text to search for
        """
        # Simple word matching for demonstration
        matching_chunks = []
        for chunk in self.chunks:
            # Count how many words from the query appear in the chunk
            query_words = set(query_text.lower().split())
            chunk_words = set(chunk["text"].lower().split())
            match_score = len(query_words.intersection(chunk_words))
            
            if match_score > 0:
                matching_chunks.append((chunk, match_score))
        
        # Sort by match score
        matching_chunks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\n" + "="*80)
        logger.info(f"QUERY: {query_text}")
        logger.info("="*80)
        
        if matching_chunks:
            # Take the top match
            top_chunk, score = matching_chunks[0]
            logger.info(f"TOP MATCHING CHUNK (Score: {score}):")
            logger.info(f"CHUNK {top_chunk['chunk_id']} (Page {top_chunk['page_num'] + 1}):")
            logger.info(f"TEXT: {top_chunk['text'][:200]}...")
            
            # Get associated images
            associated_images = self.chunk_image_map.get(top_chunk["chunk_id"], [])
            logger.info(f"ASSOCIATED IMAGES ({len(associated_images)}):")
            
            for img in associated_images:
                logger.info(f"  - {img['image_path']} (Position: {img['rect']})")
        else:
            logger.info("No matching chunks found for the query.")

    def save_chunks_to_disk(self, output_dir="chunks"):
        """Save chunks and their associated images to disk as JSON files."""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save all chunks to a single JSON file
        all_chunks_path = os.path.join(output_dir, "all_chunks.json")
        
        # Create a serializable version of chunks (remove non-serializable objects)
        serializable_chunks = []
        for chunk in self.chunks:
            # Create a copy of the chunk
            serializable_chunk = chunk.copy()
            
            # Convert rect objects to serializable dictionaries
            if serializable_chunk["start_rect"]:
                serializable_chunk["start_rect"] = {
                    "x0": serializable_chunk["start_rect"].x0,
                    "y0": serializable_chunk["start_rect"].y0,
                    "x1": serializable_chunk["start_rect"].x1,
                    "y1": serializable_chunk["start_rect"].y1
                }
            
            if serializable_chunk["end_rect"]:
                serializable_chunk["end_rect"] = {
                    "x0": serializable_chunk["end_rect"].x0,
                    "y0": serializable_chunk["end_rect"].y0,
                    "x1": serializable_chunk["end_rect"].x1,
                    "y1": serializable_chunk["end_rect"].y1
                }
            
            # Add associated images
            associated_images = self.chunk_image_map.get(chunk["chunk_id"], [])
            serializable_chunk["associated_images"] = []
            
            for img in associated_images:
                # Create a serializable version of the image info
                serializable_img = {
                    "page_num": img["page_num"],
                    "image_path": img["image_path"],
                    "rect": {
                        "x0": img["rect"].x0,
                        "y0": img["rect"].y0,
                        "x1": img["rect"].x1,
                        "y1": img["rect"].y1
                    } if img["rect"] else None,
                    "xref": img["xref"]
                }
                serializable_chunk["associated_images"].append(serializable_img)
            
            serializable_chunks.append(serializable_chunk)
        
        # Save to file
        with open(all_chunks_path, "w") as f:
            json.dump(serializable_chunks, f, indent=2)
        
        logger.info(f"Saved all chunks to {all_chunks_path}")
        
        # Save each chunk with its associated images to individual files
        for chunk in serializable_chunks:
            chunk_id = chunk["chunk_id"]
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_id}.json")
            
            with open(chunk_path, "w") as f:
                json.dump(chunk, f, indent=2)
            
            logger.info(f"Saved chunk {chunk_id} to {chunk_path}")
        
        # Create a sample file with a few chunks for quick examination
        sample_chunks = serializable_chunks[:5]  # First 5 chunks
        sample_path = os.path.join(output_dir, "sample_chunks.json")
        
        with open(sample_path, "w") as f:
            json.dump(sample_chunks, f, indent=2)
        
        logger.info(f"Saved sample chunks to {sample_path}")
        
        return serializable_chunks


def main():
    """Process the PDF and extract text chunks and images."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a PDF file to extract text and images.")
    parser.add_argument("--pdf", default="20.072.30.03EN.pdf", help="Path to the PDF file to process")
    args = parser.parse_args()
    
    pdf_path = args.pdf
    
    # Create extractor
    extractor = PDFTextImageExtractor(pdf_path=pdf_path, output_dir="extracted_images")
    
    # Process PDF
    extractor.process_pdf()
    
    # Save chunks to disk
    extractor.save_chunks_to_disk(output_dir="chunks")
    
    # Test with some sample queries
    logger.info("Testing retrieval with sample queries...")
    test_queries = [
        "safety instructions",
        "maintenance procedures",
        "installation requirements",
        "explosive atmospheres",
        "agitator assembly"
    ]
    
    for query in test_queries:
        extractor.simulate_retrieval(query)
    
    logger.info("PDF processing complete!")


if __name__ == "__main__":
    main() 