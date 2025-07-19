import os
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import hashlib
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a processed chunk of document content"""
    content: str
    chunk_id: str
    page_number: int
    chunk_type: str  # 'text', 'table', 'equation', 'figure'
    metadata: Dict[str, Any]
    embeddings: List[float] = None


class ScientificDocumentProcessor:
    """Advanced processor for scientific documents with equations, tables, and figures"""

    def __init__(self, output_dir: str = "processed_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize models
        self.table_extractor = pipeline(
            "table-question-answering", model="microsoft/table-transformer-detection")
        self.equation_detector = self._load_equation_detector()

    def _load_equation_detector(self):
        """Load equation detection model"""
        try:
            return pipeline("image-classification", model="microsoft/layoutlm-base-uncased")
        except:
            logger.warning(
                "Equation detection model not available, using regex fallback")
            return None

    def process_document(self, pdf_path: str) -> List[DocumentChunk]:
        """Process a single PDF document"""
        logger.info(f"Processing document: {pdf_path}")

        doc = fitz.open(pdf_path)
        chunks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract text
            text_chunks = self._extract_text_chunks(page, page_num)
            chunks.extend(text_chunks)

            # Extract tables
            table_chunks = self._extract_tables(page, page_num)
            chunks.extend(table_chunks)

            # Extract equations
            equation_chunks = self._extract_equations(page, page_num)
            chunks.extend(equation_chunks)

            # Extract figures
            figure_chunks = self._extract_figures(page, page_num)
            chunks.extend(figure_chunks)

        doc.close()
        return chunks

    def _extract_text_chunks(self, page, page_num: int) -> List[DocumentChunk]:
        """Extract and chunk text content"""
        text = page.get_text()

        # Split into semantic chunks (sections, paragraphs)
        chunks = self._semantic_text_chunking(text)

        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue

            chunk_id = f"text_{page_num}_{i}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"

            document_chunks.append(DocumentChunk(
                content=chunk_text.strip(),
                chunk_id=chunk_id,
                page_number=page_num,
                chunk_type="text",
                metadata={
                    "source_page": page_num,
                    "chunk_index": i,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text)
                }
            ))

        return document_chunks

    def _semantic_text_chunking(self, text: str) -> List[str]:
        """Intelligent text chunking based on semantic boundaries"""
        # Split by headers (common in scientific papers)
        header_patterns = [
            r'\n\d+\.\s+[A-Z][^.\n]*\n',  # Numbered sections
            r'\n[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'\n\d+\.\d+\s+[A-Z][^.\n]*\n',  # Subsections
        ]

        chunks = [text]
        for pattern in header_patterns:
            new_chunks = []
            for chunk in chunks:
                split_chunks = re.split(pattern, chunk)
                new_chunks.extend(split_chunks)
            chunks = new_chunks

        # Further split by paragraphs
        final_chunks = []
        for chunk in chunks:
            paragraphs = chunk.split('\n\n')
            final_chunks.extend([p.strip() for p in paragraphs if p.strip()])

        return final_chunks

    def _extract_tables(self, page, page_num: int) -> List[DocumentChunk]:
        """Extract tables from the page"""
        tables = page.find_tables()
        chunks = []

        for i, table in enumerate(tables):
            try:
                # Convert table to structured format
                table_data = table.extract()
                table_df = pd.DataFrame(table_data[1:], columns=table_data[0])

                # Create table description
                table_description = self._describe_table(table_df)

                chunk_id = f"table_{page_num}_{i}_{hashlib.md5(table_description.encode()).hexdigest()[:8]}"

                # Only keep primitive metadata (remove 'table_data')
                chunks.append(DocumentChunk(
                    content=table_description,
                    chunk_id=chunk_id,
                    page_number=page_num,
                    chunk_type="table",
                    metadata={
                        "source_page": page_num,
                        "table_index": i,
                        "rows": len(table_df),
                        "columns": len(table_df.columns)
                        # 'table_data' removed to ensure all metadata values are primitives
                    }
                ))
            except Exception as e:
                logger.warning(
                    f"Failed to extract table {i} on page {page_num}: {e}")

        return chunks

    def _describe_table(self, df: pd.DataFrame) -> str:
        """Generate a natural language description of the table"""
        description = f"Table with {len(df)} rows and {len(df.columns)} columns. "
        description += f"Columns: {', '.join(df.columns)}. "

        # Add sample data
        if len(df) > 0:
            sample_row = df.iloc[0].to_dict()
            description += f"Sample data: {sample_row}"

        return description

    def _extract_equations(self, page, page_num: int) -> List[DocumentChunk]:
        """Extract equations from the page"""
        # Look for LaTeX-style equations
        text = page.get_text()
        equation_patterns = [
            r'\$[^$]+\$',  # Inline equations
            r'\$\$[^$]+\$\$',  # Display equations
            # Equation environments
            r'\\begin\{equation\}.*?\\end\{equation\}',
        ]

        chunks = []
        for pattern in equation_patterns:
            equations = re.findall(pattern, text, re.DOTALL)
            for i, eq in enumerate(equations):
                chunk_id = f"equation_{page_num}_{i}_{hashlib.md5(eq.encode()).hexdigest()[:8]}"

                chunks.append(DocumentChunk(
                    content=f"Equation: {eq}",
                    chunk_id=chunk_id,
                    page_number=page_num,
                    chunk_type="equation",
                    metadata={
                        "source_page": page_num,
                        "equation_index": i,
                        "equation_type": "latex",
                        "raw_equation": eq
                    }
                ))

        return chunks

    def _extract_figures(self, page, page_num: int) -> List[DocumentChunk]:
        """Extract figures and their captions"""
        # Get image list
        image_list = page.get_images()
        chunks = []

        for i, img in enumerate(image_list):
            try:
                # Get image
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                # Look for caption near the image
                caption = self._find_figure_caption(page, img)

                chunk_id = f"figure_{page_num}_{i}_{hashlib.md5(caption.encode()).hexdigest()[:8]}"

                chunks.append(DocumentChunk(
                    content=f"Figure: {caption}",
                    chunk_id=chunk_id,
                    page_number=page_num,
                    chunk_type="figure",
                    metadata={
                        "source_page": page_num,
                        "figure_index": i,
                        "caption": caption,
                        "image_width": pix.width,
                        "image_height": pix.height
                    }
                ))

                pix = None  # Free memory
            except Exception as e:
                logger.warning(
                    f"Failed to extract figure {i} on page {page_num}: {e}")

        return chunks

    def _find_figure_caption(self, page, img) -> str:
        """Find caption text near the image"""
        # This is a simplified approach - in practice, you'd use more sophisticated
        # layout analysis to find captions
        text = page.get_text()
        return f"Figure on page {page.number + 1}"

    def process_all_documents(self, pdf_directory: str) -> Dict[str, List[DocumentChunk]]:
        """Process all PDF documents in a directory"""
        pdf_dir = Path(pdf_directory)
        all_chunks = {}

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            try:
                chunks = self.process_document(str(pdf_file))
                all_chunks[pdf_file.stem] = chunks
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")

        return all_chunks

    def save_processed_data(self, all_chunks: Dict[str, List[DocumentChunk]], output_file: str):
        """Save processed chunks to JSONL file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc_name, chunks in all_chunks.items():
                for chunk in chunks:
                    # Convert to JSON-serializable format
                    chunk_dict = {
                        "document": doc_name,
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "chunk_type": chunk.chunk_type,
                        "metadata": chunk.metadata
                    }
                    f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

        logger.info(f"Saved processed data to {output_file}")


def main():
    """Main processing function"""
    processor = ScientificDocumentProcessor()

    # Process all documents
    pdf_directory = "documents"  # Directory containing your 57 PDFs
    all_chunks = processor.process_all_documents(pdf_directory)

    # Save to JSONL
    output_file = "belle2_processed_chunks.jsonl"
    processor.save_processed_data(all_chunks, output_file)

    # Print statistics
    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    print(f"Processing complete!")
    print(f"Documents processed: {len(all_chunks)}")
    print(f"Total chunks created: {total_chunks}")

    # Chunk type breakdown
    chunk_types = {}
    for chunks in all_chunks.values():
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(
                chunk.chunk_type, 0) + 1

    print("\nChunk type breakdown:")
    for chunk_type, count in chunk_types.items():
        print(f"  {chunk_type}: {count}")


if __name__ == "__main__":
    main()
