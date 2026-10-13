# app/main.py
from fastapi import FastAPI,HTTPException
from langchain_groq import ChatGroq
from contextlib import asynccontextmanager
from services.ingestion import create_vector_store
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
vector_store = None
groq_llm_model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
            streaming=True
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    vector_store = create_vector_store()
    yield

app = FastAPI(
                title="DocuMind AI",
                description="Ask questions about your PDF documents",
                lifespan=lifespan
            )

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AnswerResponse(BaseModel):
    question: str
    answer: str


def answer_question(query: str) -> str:
    retrieved_docs = vector_store.similarity_search(query, k=4)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_template("""
    You are DocuMind AI, a PDF question-answering assistant.
    Answer using only the context below.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """)

    response = (prompt | groq_llm_model).invoke({
        "context": context,
        "question": query,
    })

    return response.content

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = answer_question(request.question)

        return AnswerResponse(
            question=request.question,
            answer=answer,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
        ) from exc