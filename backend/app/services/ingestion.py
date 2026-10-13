from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore




script_dir = Path(__file__).parent 
#print(script_dir)
backend_dir = script_dir.parent.parent 
#print(backend_dir)
pdf_path = backend_dir / "data" / "data_science_syllabus.pdf"
#print(f"Path to PDF: {pdf_path}")




def load_pdf():
    documents = []
    # 1. Load the PDF directly using pypdf 
    pdf_reader =  PdfReader(pdf_path)
    for page_num, page in enumerate(pdf_reader.pages):
        documents.append(
            Document(
                page_content=page.extract_text(),
                metadata = {
                    "source": pdf_path.name,
                    "page": page_num,
                }
            )
        )

    return documents

def create_vector_store():
    documents = load_pdf()
    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = pdf_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
    vector_store = InMemoryVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
    print(f"Loaded {len(documents)} pages")
    print(f"Created {len(chunks)} chunks")

    return vector_store



