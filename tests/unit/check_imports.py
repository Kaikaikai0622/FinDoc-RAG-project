import sys
import traceback

sys.path.insert(0, r"/")

tests = [
    ("plain_text_parser", "from src.ingestion.plain_text_parser import PlainTextParser"),
    ("xlsx_parser", "from src.ingestion.xlsx_parser import XlsxParser"),
    ("docx_parser", "from src.ingestion.docx_parser import DocxParser"),
    ("pptx_parser", "from src.ingestion.pptx_parser import PptxParser"),
    ("document_router", "from src.ingestion.document_router import DocumentRouter, UnsupportedFileTypeError"),
    ("pipeline", "from src.ingestion.pipeline import IngestionPipeline"),
    ("__init__", "from src.ingestion import DocumentRouter, PlainTextParser"),
]

for name, stmt in tests:
    try:
        exec(stmt)
        print(f"OK  {name}")
    except Exception as e:
        print(f"ERR {name}: {e}")
        traceback.print_exc()

