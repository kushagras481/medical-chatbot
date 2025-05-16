import os
import textwrap
import warnings
from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────────────────
# Obfuscator Module: JSON-based two-step PHI obfuscation
# ──────────────────────────────────────────

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment. Please add it to your .env file.")

# Initialize Gemini
warnings.filterwarnings("ignore")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def configure_gemini() -> bool:
    """
    Ensure Gemini API is configured.
    Returns True if configured successfully.
    """
    if not API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment.")
        return False
    genai.configure(api_key=API_KEY)
    return True


def extract_phi_json(original: str) -> str:
    """
    Extract PHI spans in the input text as JSON mapping entity types to
    character index spans. Returns only the JSON string.
    """
    few_shot = textwrap.dedent("""
        # Examples of PHI extraction
        Text: Patient Jane Smith, ID 123-45-6789, was admitted on 2025-04-01 for acute bronchitis.
        JSON: {"patient_name": [[8, 18]], "ssn": [[23, 34]], "admission_date": [[45, 55]], "diagnosis": [[60, 74]]}
        ---
        Text: Dr. Alice Lee prescribed 5mg of Lisinopril to patient Bob Brown (DOB 02/02/1970).
        JSON: {"doctor_name": [[4, 15]], "dosage": [[27, 31]], "medication": [[35, 45]], "patient_name": [[49, 59]], "dob": [[65, 75]]}
        ---
        Text: Subject ID: AB-1234 visited on 2025-03-15 complaining of headache and nausea.
        JSON: {"subject_id": [[12, 19]], "visit_date": [[31, 41]], "symptoms": [[55, 68]]}
        ---
        Text: I\'m Doyoon Kim. I wanna know how to be happy
        JSON: {"person_name": [[4, 15]]}
        ---
    """)
    prompt = textwrap.dedent(f"""
        {few_shot}
        Identify all personal health information (PHI) in the text below and
        output **only** valid JSON mapping each entity type to a list of
        [start, end] character indices.
        Entity types to label: patient_name, doctor_name, person_name, ssn, dob, admission_date, visit_date, diagnosis, medication, dosage, symptoms.
        Use the most specific type possible.

        Text: {original}
        JSON:
    """)
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    # Remove any surrounding markdown or backticks
    return text.lstrip('`\n ').rstrip('`\n ')


def obfuscate_from_json(original: str, phi_json: str) -> str:
    """
    Given the original text and a PHI JSON mapping, replace each PHI span with
    an uppercase placeholder <ENTITY_TYPE> and return the obfuscated text.
    """
    prompt = textwrap.dedent(f"""
        Original: {original}
        PHI: {phi_json}

        Instructions:
        1. For each entry in the JSON, replace the exact span in the Original text
           with the placeholder <ENTITY_TYPE>, where ENTITY_TYPE is the uppercase key.
        2. Preserve all other text exactly as is.
        3. Do not output any JSON or commentary—only the final obfuscated text.

        Obfuscated:
    """)
    resp = model.generate_content(prompt)
    return resp.text.strip()


def obfuscate_text(original: str, debug: bool = False) -> str:
    """
    Perform the full two-step obfuscation pipeline:
    1) extract PHI JSON from the original text
    2) obfuscate the original using that JSON

    If debug=True, prints original, JSON, and obfuscated text.
    Returns the obfuscated text.
    """
    if not configure_gemini():
        raise RuntimeError("Gemini API not configured")

    phi_json = extract_phi_json(original)
    obfuscated = obfuscate_from_json(original, phi_json)

    if 1:
        print("=== ORIGINAL TEXT ===")
        print(original)
        print("\n=== EXTRACTED JSON ===")
        print(phi_json)
        print("\n=== OBFUSCATED TEXT ===")
        print(obfuscated)

    return obfuscated


if __name__ == "__main__":
    # Simple CLI for testing with full visibility
    import argparse
    parser = argparse.ArgumentParser(
        description="Two-step PHI obfuscation (JSON→Obfuscate) with debug output"
    )
    parser.add_argument("text", help="Original text to obfuscate")
    parser.add_argument("--debug", action="store_true", help="Print intermediate JSON and debug info")
    args = parser.parse_args()

    obfuscate_text(args.text, debug=args.debug)
