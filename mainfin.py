from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import os
import PyPDF2
import requests
import tempfile
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Document Parser API", version="1.0.0")

# Security
security = HTTPBearer()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCdv_P2R4sjaVy_Cd5fQ3GROjYzZJfD4FI")
API_KEY = os.getenv("API_KEY", "your-api-key-here")  # Set your API key

# Pydantic models
class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

# Global variables for loaded models
embedding_model = None
layer1_hybrid_retriever = None
llm = None

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key from the Authorization header"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.on_event("startup")
async def startup_event():
    """Initialize models and load Layer 1 vector store on startup"""
    global embedding_model, layer1_hybrid_retriever, llm
    
    try:
        logger.info("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info("Loading Layer 1 vector store...")
        if os.path.exists("faiss_layer1_db"):
            db = FAISS.load_local("faiss_layer1_db", embedding_model, allow_dangerous_deserialization=True)
            
            # Create FAISS retriever
            faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7
            })
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(list(db.docstore._dict.values()))
            bm25_retriever.k = 6
            
            # Create hybrid ensemble retriever
            layer1_hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
            logger.info("Layer 1 hybrid retriever initialized successfully")
        else:
            logger.warning("faiss_layer1_db not found. Please ensure it exists in the working directory.")
        
        logger.info("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
        
        logger.info("Startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return local file path"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        all_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
        return all_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def create_layer2_retriever(text: str):
    """Create Layer 2 retriever from document text"""
    try:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        docs = splitter.create_documents([text])
        
        # Create FAISS vector store for Layer 2
        policy_db = FAISS.from_documents(docs, embedding_model)
        
        # Create retriever
        layer2_retriever = policy_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        
        return layer2_retriever
    except Exception as e:
        logger.error(f"Error creating Layer 2 retriever: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create Layer 2 retriever: {str(e)}")

def format_docs(docs):
    """Format documents for prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(layer2_retriever):
    """Create the RAG chain for question answering"""
    try:
        # Query refinement template
        refine_query_template = """
        Based on the initial context from a general insurance database, refine the original question to be more specific.
        This refined question will be used to find precise details in a new, specific policy document.
        Only output the new, refined question.

        Original Question: {question}

        Initial Context from General Database:
        ---
        {context}
        ---

        Refined Question for the specific policy document:
        """
        refine_prompt = PromptTemplate.from_template(refine_query_template)
        
        # Query refinement chain using Layer 1
        refine_query_chain = (
            {"context": layer1_hybrid_retriever | format_docs, "question": RunnablePassthrough()}
            | refine_prompt
            | llm
            | StrOutputParser()
        )
        
        # Final answering template
        final_prompt_template = """
        You are an expert insurance assistant. Answer the user's question based ONLY on the final context provided from the specific policy document.
        Be concise and clear. If the context is insufficient, state that the information is not available in the provided document.

        Final Context:
        ---
        {context}
        ---

        Question: {question}

        Answer:
        """
        final_prompt = PromptTemplate.from_template(final_prompt_template)
        
        # Final RAG chain using Layer 2
        final_rag_chain = (
            {
                "context": refine_query_chain | layer2_retriever | format_docs,
                "question": refine_query_chain
            }
            | final_prompt
            | llm
            | StrOutputParser()
        )
        
        return final_rag_chain
        
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create RAG chain: {str(e)}")

async def process_question_async(question: str, rag_chain) -> str:
    """Process a single question asynchronously"""
    try:
        # Run the chain in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, rag_chain.invoke, question)
        return answer
    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}")
        return f"Error processing question: {str(e)}"

@app.post("/hackrx/run", response_model=QuestionResponse)
async def run_rag(
    request: QuestionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process documents and answer questions using RAG pipeline
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Check if models are loaded
        if not all([embedding_model, layer1_hybrid_retriever, llm]):
            raise HTTPException(
                status_code=500, 
                detail="Models not properly initialized. Please check server logs."
            )
        
        # Download and process the document
        logger.info("Downloading document...")
        pdf_path = download_pdf_from_url(request.documents)
        
        try:
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            document_text = extract_text_from_pdf(pdf_path)
            
            # Create Layer 2 retriever
            logger.info("Creating Layer 2 retriever...")
            layer2_retriever = create_layer2_retriever(document_text)
            
            # Create RAG chain
            logger.info("Creating RAG chain...")
            rag_chain = create_rag_chain(layer2_retriever)
            
            # Process all questions
            logger.info("Processing questions...")
            answers = []
            for i, question in enumerate(request.questions):
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                answer = await process_question_async(question, rag_chain)
                answers.append(answer)
            
            logger.info("All questions processed successfully")
            return QuestionResponse(answers=answers)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_rag: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([embedding_model, layer1_hybrid_retriever, llm])
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Document Parser API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )