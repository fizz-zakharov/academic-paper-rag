import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document


def extract_blocks_reading_order(page: fitz.Page) -> List[str]:
    """
    Extracts text blocks from a PDF page and sorts them to preserve
    two-column scientific paper reading order:
    1. Full-width spans (Title, Abstract, etc.)
    2. Left column (top to bottom)
    3. Right column (top to bottom)
    """
    page_rect = page.rect
    mid_x = page_rect.width / 2.0

    # Extract raw blocks: (x0, y0, x1, y1, text, block_no, block_type)
    # block_type == 0 is text; 1 is an image/drawing
    raw_blocks = page.get_text("blocks")
    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]

    full_width_blocks = []
    left_column_blocks = []
    right_column_blocks = []

    for block in text_blocks:
        x0, y0, x1, y1, text = block[:5]
        block_width = x1 - x0

        # Blocks spanning > 60% of total page width are full-width
        if block_width > (page_rect.width * 0.60):
            full_width_blocks.append((y0, text.strip()))
        elif x0 < mid_x:
            left_column_blocks.append((y0, text.strip()))
        else:
            right_column_blocks.append((y0, text.strip()))

    # Sort each zone vertically by y0 (top to bottom)
    full_width_blocks.sort(key=lambda b: b[0])
    left_column_blocks.sort(key=lambda b: b[0])
    right_column_blocks.sort(key=lambda b: b[0])

    ordered_texts = []
    # 1. Add full-width elements first
    ordered_texts.extend([text for _, text in full_width_blocks])
    # 2. Add left column
    ordered_texts.extend([text for _, text in left_column_blocks])
    # 3. Add right column
    ordered_texts.extend([text for _, text in right_column_blocks])

    return ordered_texts


def parse_academic_pdf(file_path: str) -> List[Document]:
    """
    Parses a two-column academic PDF into a list of LangChain Document
    objects with page-level metadata.
    """
    doc = fitz.open(file_path)
    documents: List[Document] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        ordered_blocks = extract_blocks_reading_order(page)
        page_text = "\n\n".join(ordered_blocks)

        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                    },
                )
            )

    doc.close()
    return documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Parsing: {test_path}")
        docs = parse_academic_pdf(test_path)
        print(f"Total pages extracted: {len(docs)}")
        if docs:
            print("\n--- Preview Page 1 (First 400 chars) ---")
            print(docs[0].page_content[:400])
            print(f"\nMetadata: {docs[0].metadata}")
    else:
        print("Usage: python src/parser.py <path_to_pdf>")