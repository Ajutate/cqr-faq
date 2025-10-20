"""
Advanced PDF extraction module with table support
Uses pdfplumber (MIT License - completely free and open source)
"""
import pdfplumber
from typing import List, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedPDFExtractor:
    """Extract text and tables from PDFs with high accuracy using pdfplumber"""
    
    def __init__(self):
        """Initialize PDF extractor"""
        logger.info("✅ PDFplumber extractor initialized")
    
    def extract_with_pdfplumber(self, file_path: str, max_pages: int = None) -> List[Dict]:
        """
        Extract text and tables using pdfplumber (MIT License - Free!)
        Excellent for structured data and tables
        
        Args:
            file_path: Path to PDF file
            max_pages: Optional limit on number of pages to process (None = all pages)
        """
        documents = []
        
        try:
            logger.info(f"🔍 Processing PDF: {Path(file_path).name}")
            
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
                
                logger.info(f"📄 Total pages: {total_pages}, Processing: {pages_to_process}")
                
                for page_num, page in enumerate(pdf.pages[:pages_to_process]):
                    try:
                        # Extract tables first
                        tables = page.extract_tables()
                        
                        if tables:
                            for table_idx, table in enumerate(tables):
                                # Convert table to formatted text
                                table_text = self._format_table(table)
                                
                                if table_text.strip():
                                    documents.append({
                                        "content": f"TABLE {table_idx + 1} on Page {page_num + 1}:\n{table_text}",
                                        "page": page_num + 1,
                                        "source": Path(file_path).name,
                                        "method": "pdfplumber_table",
                                        "type": "table"
                                    })
                        
                        # Extract regular text
                        text = page.extract_text()
                        if text and text.strip():
                            documents.append({
                                "content": text,
                                "page": page_num + 1,
                                "source": Path(file_path).name,
                                "method": "pdfplumber_text",
                                "type": "text"
                            })
                        
                        # Log progress every 10 pages for large PDFs
                        if (page_num + 1) % 10 == 0:
                            logger.info(f"📊 Progress: {page_num + 1}/{pages_to_process} pages processed")
                    
                    except Exception as page_error:
                        logger.warning(f"⚠️ Error on page {page_num + 1}: {page_error}, skipping...")
                        continue
            
            logger.info(f"✅ Extracted {len(documents)} sections from {pages_to_process} pages")
            return documents
        
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            raise Exception(f"Failed to extract PDF: {str(e)}")
    
    def _format_table(self, table: List[List]) -> str:
        """
        Format table data into readable text
        Preserves structure and relationships
        """
        if not table:
            return ""
        
        formatted_lines = []
        
        # Determine column widths
        col_widths = []
        for col_idx in range(len(table[0]) if table else 0):
            max_width = max(
                len(str(row[col_idx] or "")) 
                for row in table 
                if col_idx < len(row)
            )
            col_widths.append(max_width)
        
        # Format each row
        for row_idx, row in enumerate(table):
            formatted_row = " | ".join(
                str(cell or "").ljust(col_widths[idx])
                for idx, cell in enumerate(row)
            )
            formatted_lines.append(formatted_row)
            
            # Add separator after header
            if row_idx == 0:
                formatted_lines.append("-" * len(formatted_row))
        
        return "\n".join(formatted_lines)
    
    def analyze_pdf(self, file_path: str) -> Dict:
        """
        Analyze PDF structure and provide extraction recommendations
        """
        info = {
            "filename": Path(file_path).name,
            "pages": 0,
            "has_tables": False,
            "has_text": False,
            "method": "pdfplumber"
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                info["pages"] = len(pdf.pages)
                
                # Check first few pages for tables
                for page in pdf.pages[:3]:
                    tables = page.extract_tables()
                    if tables:
                        info["has_tables"] = True
                    
                    text = page.extract_text()
                    if text and text.strip():
                        info["has_text"] = True
        
        except Exception as e:
            info["error"] = str(e)
        
        return info


# Helper function for easy import
def extract_pdf(file_path: str, method: str = "pdfplumber") -> List[Dict]:
    """
    Extract content from PDF using pdfplumber (MIT License - completely free!)
    
    Args:
        file_path: Path to PDF file
        method: Only 'pdfplumber' is available (free and open source)
    
    Returns:
        List of document chunks with metadata
    """
    extractor = AdvancedPDFExtractor()
    documents = extractor.extract_with_pdfplumber(file_path)
    
    return documents
