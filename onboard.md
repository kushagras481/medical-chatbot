Simple Medical Advice Chatbot Application Skeleton

Application Objective

The primary focus is to create a simple general medical advice chatbot that provides patient-friendly English responses to medical queries, deployable on hospital tablets. The chatbot uses the Gemini API for response generation, incorporates sensitive data redaction, and employs a federated learning (FL)-trained Gemini model for secure training. The application supports research on:





Federated Learning (FL): Ensures medical patient data privacy during training while maintaining response accuracy (BLEU >0.6, COMET >0.8).



Catastrophic Forgetting: Measures retention of non-medical capabilities post-FL fine-tuning (accuracy drop <10% on non-medical tasks).

The application is built with a Next.js frontend and Python backend, using few-shot learning for response accuracy and supporting catastrophic forgetting tests with non-medical queries. It handles 50 medical queries (20% rare diseases), achieves >80% user approval (3-patient study), and demonstrates 5% COMET improvement over ChatGPT.

Application Structure

1. Overview





Type: Web-based chatbot application



Scope: Single app for multiple hospital tablets, no account integrations or external APIs



Focus: Simple, user-friendly medical chatbot with privacy-preserving features

2. Components

2.1 Frontend





Technology: Next.js (v13+)



Purpose: Simple user interface for patients to input medical queries and view responses



Structure:





medical-chatbot/





pages/index.js: Main page with input form, submit button, and response display



pages/api/chat.js: API route to forward queries to Python backend



styles/: CSS (inline or Tailwind) for hospital-like styling



Features:





Input box: Free-text field for medical queries (5–50 words)



Submit button: “Ask” button to send queries



Response area: Displays patient-friendly answers from Gemini



Responsive design for tablet screens (10-inch)



Styling:





Hospital-like: White background, blue/green accents (#005555), Arial font, 16px+ text



Large buttons and text for accessibility

2.2 Backend





Technology: Python 3.8+, Flask



Purpose: Process queries, call Gemini API, integrate redaction and FL-trained model



Structure:





backend/





app.py: Flask server with Gemini API logic, few-shot prompt, redaction, and FL model integration



requirements.txt: Dependencies (flask, requests)



Features:





Accepts POST requests from frontend



Uses few-shot learning with 3 prompt examples (symptom, disease, rare disease)



Redacts sensitive data in queries



Uses FL-trained Gemini model for responses



Appends disclaimer (“Consult a doctor”) to responses

2.3 Datasets





Purpose: Support training, evaluation, and catastrophic forgetting tests



Structure:





datasets/





training/: Medical dataset (100 samples for few-shot prompts and testing)



evaluation/: Medical dataset (50 queries for BLEU/COMET)



forgetting/: Non-medical dataset (5–10 queries for forgetting tests)



Requirements:





Separate datasets for training and evaluation to avoid bias



Non-medical queries to assess catastrophic forgetting

3. Inputs and Outputs

3.1 Inputs





Medical Queries:





Format: Free-text, English, 5–50 words



Types:





Symptom Descriptions (50%): “I have a headache and feel dizzy.”



Disease Inquiries (30%, 20% rare diseases): “What is cystic fibrosis?”



General Advice (20%): “How can I stay healthy?”



Processing: Redacted for sensitive data before Gemini API call



Non-Medical Queries:





Format: Free-text, English, 5–20 words



Examples: “What is the capital of France?”, “What is photosynthesis?”



Purpose: Test catastrophic forgetting (separate script, not patient-facing)

3.2 Outputs





Format: Patient-friendly English, <100 words, middle school reading level



Content: Accurate medical advice with a disclaimer



Examples:





Input: “I have a headache and feel tired.” → Output: “Headaches and tiredness can come from stress. Rest, drink water, and consult a doctor.”



Input: “What is dyspnea?” → Output: “Dyspnea means trouble breathing. Consult a doctor.”

4. Technical Specifications

4.1 Frontend





Framework: Next.js with Axios for API calls



Styling: Inline CSS or Tailwind for responsive, hospital-like design



Environment: Node.js (v16+)



Deployment: Vercel or local server, accessible via browser on tablets

4.2 Backend





Framework: Python Flask



Dependencies: Requests for Gemini API calls



API: Gemini API (requires key and endpoint)



Integration: Redaction for sensitive data, FL-trained Gemini model

4.3 Performance





Latency: <2s per query



Resource Usage: <50% CPU/memory on tablet-like environment (tested in Colab)



Security: HTTPS for API calls, no storage of patient queries

4.4 Evaluation Metrics





Medical Queries (50, 20% rare diseases):





BLEU >0.6, COMET >0.8



User approval >80% (3-patient study)



5% COMET improvement over ChatGPT



Non-Medical Queries (5–10):





Accuracy drop <10% post-FL fine-tuning



Privacy:





Redaction accuracy >95%



FL convergence <5% accuracy drop

5. Research Support





Federated Learning (FL):





Trains Gemini across multiple nodes (100 samples) without sharing raw data



Ensures privacy and maintains response accuracy



Catastrophic Forgetting:





Tests FL-trained Gemini on 5–10 non-medical queries



Compares to baseline Gemini (pre-fine-tuning) to measure retention

6. Constraints





Beginner-friendly implementation (minimal complexity)



Single app for multiple hospital tablets (browser-based)



No account integrations or external APIs beyond Gemini



Supports 50 medical queries and 5–10 non-medical queries

7. Dependencies





External:





Gemini API key and endpoint



Google Colab for development and testing



BLEU (sacrebleu) and COMET (Hugging Face) tools



Datasets:





Training: Medical dataset (100 samples)



Evaluation: Medical dataset (50 queries)



Forgetting: Non-medical dataset (5–10 queries)