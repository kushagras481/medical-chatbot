# Medical Chatbot

A simple medical advice chatbot with Next.js frontend, FastAPI intermediary, and Python/Gemini backend.

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
- Uvicorn ASGI server

## Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python 3.12 (Python 3.13 is not currently supported due to dependency issues)
- Google Gemini API key

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

4. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   PORT=8000
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
- Simple and accessible chat interface
- Hospital-friendly styling with medical-themed color scheme
- Sensitive data redaction for privacy
- Few-shot learning prompts for better medical responses
- Responsive design for mobile and desktop
- Real-time chat with loading states
- Error handling for API failures

## Usage

1. Type a medical question in the input box
2. Click "Ask" or press Enter
3. The AI will generate a patient-friendly response
4. Continue the conversation as needed

## API Flow

1. User sends query through the frontend
2. Frontend sends POST request to `/api/chat` endpoint
3. Backend processes request and sends to Gemini API
4. Gemini generates response with medical few-shot prompting
5. Response is returned to frontend and displayed to user

## Backend API Endpoints

- `GET /` - Server health check
- `GET /api/health` - Application health check
- `POST /api/chat` - Process medical queries with JSON payload `{query: string}`

## Troubleshooting

- If you encounter errors when creating the virtual environment with `python3.12 -m venv venv`, try the alternative virtualenv method shown in the Backend Setup section.
- If you encounter the error "Building wheel for pydantic-core did not run successfully", make sure you're using Python 3.12, not 3.13.
- If the frontend can't connect to the backend, check that your `.env.local` has the correct API URL and the backend server is running.
- For Gemini API errors, verify your API key is valid and properly set in the backend `.env` file. 