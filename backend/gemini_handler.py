import os
import google.generativeai as genai
from dotenv import load_dotenv
from rag import answer_medical_question
from obfuscator import obfuscate_text  

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY", "")

def configure_gemini():
    """Configure the Gemini API with the API key."""
    if not API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment.")
        return False
    
    genai.configure(api_key=API_KEY)
    return True

def redact_sensitive_data(query):
    """
    Redact potentially sensitive information from the query.
    """
    sensitive_terms = ["SSN", "social security", "address", "phone", "email"]
    redacted_query = query
    
    for term in sensitive_terms:
        if term.lower() in query.lower():
            redacted_query = redacted_query.replace(term, "[REDACTED]")
    
    return redacted_query

# def get_few_shot_prompt(query):
#     """
#     Create a few-shot prompt with medical examples to guide the model.
#     """
#     few_shot_prompt = """
#     You are a simple medical chatbot providing helpful, patient-friendly answers to diverse medical questions, including symptoms, diseases (20% rare), and general health advice. 
#     Always give concise responses (<100 words) at a middle school reading level. Ensure accuracy, even for redacted queries (e.g., "[REDACTED] has a headache"). 
#     For vague or non-medical queries, provide a general response and disclaimer. Add a disclaimer to consult a doctor for professional medical advice.
#     If the query is not medical, provide a general response.

# Examples:
#     Question: I have a headache and feel dizzy.
#     Answer: Headaches with dizziness can come from dehydration, stress, or lack of sleep. Drink water, rest in a quiet, dark room, and get enough sleep. If symptoms get worse or last long, consult a doctor for professional medical advice.

#     Question: What is diabetes?
#     Answer: Diabetes is when your body can't manage blood sugar well. It either doesn't make enough insulin or can't use it properly, causing high sugar levels. This can harm your health over time. Treatment includes healthy eating, exercise, and sometimes medicine. Consult a doctor for diagnosis and treatment.

#     Question: What is Ehlers-Danlos syndrome?
#     Answer: Ehlers-Danlos syndrome affects tissues supporting skin, joints, and organs. It can cause very flexible joints and stretchy, fragile skin. Symptoms vary, and there's no cure. Treatment helps manage symptoms and prevent problems. Consult a doctor for proper diagnosis and care.

#     Question: I have chest pain and shortness of breath. What's wrong?
#     Answer: Chest pain and shortness of breath can be serious, possibly from heart or lung issues. Rest and stay calm, but don't ignore it. Call emergency services or consult a doctor immediately for professional medical advice.

#     Question: Why am I always tired?
#     Answer: Feeling tired all the time can come from poor sleep, stress, low iron, or other health issues. Try sleeping 7–8 hours, eating healthy, and staying active. If tiredness continues, consult a doctor for professional medical advice.

#     Question: What is Marfan syndrome?
#     Answer: Marfan syndrome is a rare condition affecting connective tissues, causing long limbs, heart issues, and flexible joints. It varies in severity. Treatment focuses on managing symptoms, especially heart problems. Consult a doctor for diagnosis and care.

#     Question: How can I stay healthy during flu season?
#     Answer: To stay healthy during flu season, wash hands often, eat fruits and vegetables, get a flu shot, and sleep well. Avoid touching your face and stay away from sick people. If you feel sick, consult a doctor for professional medical advice.

#     Question: [REDACTED] feels sad and hopeless. What's wrong?
#     Answer: Feeling sad and hopeless can be signs of depression or stress. Talking to a trusted person, staying active, or seeking therapy can help. If feelings last or worsen, consult a doctor or mental health professional for advice.

#     Question: What is Huntington's disease?
#     Answer: Huntington's disease is a rare genetic disorder causing brain cell damage. It leads to movement problems, memory issues, and mood changes. There's no cure, but treatment can help manage symptoms. Consult a doctor for diagnosis and support.

#     Question: What is the capital of France?
#     Answer: The capital of France is Paris.

#     Now, please answer this medical question in a helpful, accurate, and patient-friendly way:
#     {query}
#     """
    
#     return few_shot_prompt.format(query=query)

# async def generate_medical_response(query):
#     """
#     Generate a medical response using the Gemini API.
#     """
#     # Check if Gemini is configured
#     if not configure_gemini():
#         return {
#             "response": "Sorry, the medical assistant is currently unavailable. Please try again later.",
#             "error": "API key not configured"
#         }
    
#     try:
#         # Redact sensitive information
#         redacted_query = redact_sensitive_data(query)
        
#         # Create the prompt with few-shot examples
#         prompt = get_few_shot_prompt(redacted_query)
        
#         # Get model - using Gemini Pro for general text capabilities
#         model = genai.GenerativeModel("gemini-2.0-flash")
        
#         # Generate response
#         response = model.generate_content(prompt)
        
#         return {
#             "response": response.text,
#             "error": None
#         }
    
#     except Exception as e:
#         error_msg = str(e)
#         print(f"Error generating response: {error_msg}")
#         return {
#             "response": "Sorry, I couldn't process your request. Please try again later.",
#             "error": error_msg
#         } 

async def generate_medical_response(query):
    """
    Generate a medical response using the Gemini API.
    """
    # Check if Gemini is configured
    if not configure_gemini():
        return {
            "response": "Sorry, the medical assistant is currently unavailable. Please try again later.",
            "references": None,
            "query": None,
            "error": "API key not configured"
        }
    
    try:
        # Redact sensitive information
        redacted_query = obfuscate_text(query)
        
        # Call into rag.py
        result = answer_medical_question(
            question=redacted_query,
            max_abstracts=30,
            iterations=1
        )
        return {
            "response": result["answer"],
            "references": result["references"],
            "query": result["query"],
            "error": None
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating response: {error_msg}")
        return {
            "response": "Sorry, I couldn't process your request. Please try again later.",
            "references": None,
            "query": None,
            "error": error_msg
        } 