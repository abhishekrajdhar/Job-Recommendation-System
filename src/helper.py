import fitz # PyMuPDF
import os 
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)


client = genai.GenerativeModel()


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    
    Args:
        uploaded_file (str): The path to the PDF file.
        
    Returns:
        str: The extracted text.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text



def ask_genai(prompt, model_name="gemini-2.0-flash", temperature=0.5, max_tokens=500):
    """
    Sends a prompt to the Gemini (Google GenAI) model and returns the response.

    Args:
        prompt (str): The input text prompt.
        model_name (str): Gemini model to use. Default is 'gemini-pro'.
        temperature (float): Controls randomness in response (0.0 - 1.0).
        max_tokens (int): Maximum number of output tokens.

    Returns:
        str: The generated response text.
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    )
    return response.text

