# Medical Chatbot

A sophisticated medical advice chatbot with Next.js frontend, FastAPI intermediary, and Python/Gemini backend. Features Retrieval-Augmented Generation (RAG) with PubMed integration for evidence-based responses.

## Project Structure

- `frontend/`: Next.js application
  - Uses React 19 with TypeScript
  - TailwindCSS for styling
  - Simple and accessible chat interface
  - Responsive design with hospital-friendly styling
- `backend/`: Python FastAPI server and Gemini integration
  - FastAPI for API endpoints
  - Google Gemini 2.0 Flash model integration
  - Few-shot learning prompts for better medical responses
  - PubMed RAG integration for evidence-based answers
  - Citation verification and self-feedback mechanisms
  - Sensitive data redaction

## Tech Stack

### Frontend
- Next.js 15.3.1
- React 19
- TypeScript
- TailwindCSS 4
- Geist font from Vercel

### Backend
- Python 3.12
- FastAPI
- Google Generative AI (Gemini 2.0 Flash)
- Biopython for PubMed integration
- RAG (Retrieval-Augmented Generation) pipeline
- Uvicorn ASGI server

## Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python 3.12 (Python 3.13 is not currently supported due to dependency issues)
- Google Gemini API key
- NCBI/PubMed account (optional, for higher rate limits)

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```
   # Method 1: Using venv (standard library)
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # If Method 1 fails with ensurepip errors, try Method 2:
   # Method 2: Using virtualenv (may need to install first)
   # pip install --user virtualenv
   # virtualenv venv -p python3.12
   # source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ENTREZ_EMAIL=your_email@example.com  # For PubMed API
   PORT=8000
   DEBUG=0  # Set to 1 for detailed RAG logs
   ```

5. Start the FastAPI server:
   ```
   python main.py
   ```

   The backend server will run at http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. Start the development server:
   ```
   npm run dev
   ```

   The frontend will be available at http://localhost:3000

## Features

- Medical advice chatbot using Google Gemini API
- PubMed-based RAG for evidence-based medical responses
- Citation verification and self-feedback mechanisms
- Citation Effectiveness Ratio (CER) monitoring
- Simple and accessible chat interface
- Hospital-friendly styling with medical-themed color scheme
- Sensitive data redaction for privacy
- Few-shot learning prompts for better medical responses
- Responsive design for mobile and desktop
- Real-time chat with loading states
- Error handling for API failures

## RAG Architecture

The chatbot uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline:

1. User query is transformed into an effective PubMed search query
2. Relevant medical abstracts are retrieved from PubMed
3. Gemini model generates an answer with inline citations
4. Self-feedback mechanism improves citation coverage (multiple iterations)
5. Citation verification ensures accurate references
6. Automatic answer regeneration when invalid citations are detected
7. Citation Effectiveness Ratio (CER) monitoring for quality control

## Usage

1. Type a medical question in the input box
2. Click "Ask" or press Enter
3. The AI generates an evidence-based response with PubMed citations
4. Continue the conversation as needed

## API Flow

1. User sends query through the frontend
2. Frontend sends POST request to `/api/chat` endpoint
3. Backend processes request through the RAG pipeline
4. PubMed abstracts are retrieved and processed
5. Gemini generates response with citations to medical literature
6. Response is returned to frontend and displayed to user

## Backend API Endpoints

- `GET /` - Server health check
- `GET /api/health` - Application health check
- `POST /api/chat` - Process medical queries with JSON payload `{query: string}`

## Troubleshooting

- If you encounter errors when creating the virtual environment with `python3.12 -m venv venv`, try the alternative virtualenv method shown in the Backend Setup section.
- If you encounter the error "Building wheel for pydantic-core did not run successfully", make sure you're using Python 3.12, not 3.13.
- If the frontend can't connect to the backend, check that your `.env.local` has the correct API URL and the backend server is running.
- For Gemini API errors, verify your API key is valid and properly set in the backend `.env` file.
- For PubMed integration issues, ensure you've set `ENTREZ_EMAIL` in your `.env` file.
- If RAG responses are slow, check your internet connection as PubMed API calls require network access. 