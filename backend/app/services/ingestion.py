from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate


#load ENV
load_dotenv()

script_dir = Path(__file__).parent 
print(script_dir)
backend_dir = script_dir.parent.parent 
print(backend_dir)
pdf_path = backend_dir / "data" / "data_science_syllabus.pdf"
print(f"Path to PDF: {pdf_path}")



pdf_text = []

def load_pdf():
    # 1. Load the PDF directly using pypdf 
    pdf_reader =  PdfReader(pdf_path)
    for page_num, page in enumerate(pdf_reader.pages):
        pdf_text.append(Document(
            page_content=page.extract_text(),
            metadata={"source": pdf_path, "page": page_num + 1}
        ))

def pdf_chunks(documents):

    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = pdf_splitter.split_documents(documents)
    # pdf_chunks = [info.page_content for info in data_chunks]
    # print(f"Successfully created {len(pdf_chunks)} chunks.")
    return chunks

load_pdf()
chunks = pdf_chunks(pdf_text)


#Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Vector
# vectors = embeddings.embed_documents(chunks)
# print(vectors)


#Vector Store
vector_store = InMemoryVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

# same_record = vector_store.similarity_search(query="Module 0")
# print(f"SAME - RECORD {same_record}")

#Load LLM Model
groq_llm_model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
            streaming=True
)


def answer_question(query: str):
    # 1. Search for relevant PDF chunks
    retrieved_docs = vector_store.similarity_search(query, k=4)

    # 2. Combine the retrieved text into context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # 3. Send question + context to Groq
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the context below.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """)

    response = (prompt | groq_llm_model).invoke({
        "context": context,
        "question": query
    })

    return response.content


while True:
    query = input("User: ")

    if query.lower().strip() in ["exit", "bye"]:
        break

    answer = answer_question(query)
    print("\nDocuMind AI:", answer)




