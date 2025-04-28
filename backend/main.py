from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from dotenv import load_dotenv

# Import the Gemini handler
from gemini_handler import generate_medical_response

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Medical Chatbot API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ChatRequest(BaseModel):
    query: str
    
class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests.
    Receives a query from the frontend, processes it through Gemini,
    and returns the response.
    """
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Get response from Gemini
    result = await generate_medical_response(request.query)
    
    return ChatResponse(
        response=result["response"],
        error=result["error"]
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is functioning."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 