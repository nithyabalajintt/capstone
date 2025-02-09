from typing import Dict
from typing_extensions import TypedDict
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from langdetect import detect
from deep_translator import GoogleTranslator
 
# Azure API Configuration
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"
TOKEN = "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"
 
client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
 
# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "zh-cn": "Mandarin",
    "zh-tw": "Cantonese",
    "tl": "Tagalog",
    "pa": "Punjabi",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese"
}
 
def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        detected_lang = detect(text)
        return detected_lang if detected_lang in SUPPORTED_LANGUAGES else "en"
    except:
        return "en"  # Default to English if detection fails
 
def translate_text(text: str, target_language: str = "en") -> str:
    """Translate text to the target language using deep_translator."""
    try:
        translation = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails
 
class MultilingualState(TypedDict):
    user_message: str
    translated_message: str
    detected_language: str
 
def process_multilingual_message(state: MultilingualState) -> Dict[str, str]:
    user_message = state["user_message"]
    
    detected_language = detect_language(user_message)
    
    if detected_language != "en":
        translated_message = translate_text(user_message, "en")
    else:
        translated_message = user_message
    
    return {
        "translated_message": translated_message,
        "detected_language": SUPPORTED_LANGUAGES.get(detected_language, "Unknown")
    }
 
if __name__ == "__main__":
    user_input = "Bonjour, comment ça va?"
    state = {"user_message": user_input, "translated_message": "", "detected_language": ""}
    result = process_multilingual_message(state)
    print("Detected Language:", result["detected_language"])
    print("Translated Message:", result["translated_message"]) 