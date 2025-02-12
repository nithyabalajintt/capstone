import json
from typing import Dict
import faiss  
import torch
import transformers
import langchain
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from nemoguardrails import LLMRails
from langgraph.graph import StateGraph
# Azure AI Model Configuration (Using GPT-Mini-4o)
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-mini-4o"
TOKEN = "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"
 
# Initialize Azure Client
client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
 
# Initialize Guardrails with YAML config
rails = LLMRails.from_config("config.yml")
 
# FAISS Index for Fast Lookup
index = faiss.IndexFlatL2(128)  # Example: 128-dimensional vectors
 
 
def detect_language(text: str) -> str:
    """Detects if the input language is English or Spanish."""
    lang = rails.detect_language(text)
    if lang not in ["en", "es"]:
        return "unsupported"
    return lang
 
 
def check_guardrails(user_input: str) -> Dict[str, str]:
    """Validates user input using different checks: harmful, sensitive, profanity, spam, very short, and language."""
    print("[INFO] Running Guardrails validation...")
 
    # 1. Language Detection
    lang = detect_language(user_input)
    if lang == "unsupported":
        return {
            "status": "error",
            "reason": "language_not_supported",
            "message": "We currently support only English and Spanish."
        }
 
    # 2. Process input with Azure AI Model
    messages = [SystemMessage(content="Validate this user input"), UserMessage(content=user_input)]
    response = client.complete(model=MODEL_NAME, messages=messages)
    content = response.choices[0].message.content.strip().lower()
 
    # 3. Guardrails-based filtering
    if "harmful" in content:
        return {
            "status": "error",
            "reason": "harmful_content",
            "message": "This request cannot be processed due to harmful content."
        }
    elif "sensitive" in content:
        return {
            "status": "error",
            "reason": "sensitive_content",
            "message": "This request cannot be processed due to sensitive information."
        }
    elif "profanity" in content:
        return {
            "status": "error",
            "reason": "profanity",
            "message": "Please avoid using inappropriate language."
        }
    elif "spam" in content:
        return {
            "status": "error",
            "reason": "spam_detected",
            "message": "Your message appears to be spam."
        }
    elif len(user_input.strip()) < 5:
        return {
            "status": "error",
            "reason": "short_message",
            "message": "Your message is too short. Please provide more details."
        }
 
    return {
        "status": "success",
        "reason": "validated",
        "message": "Your request is valid and being processed."
    }
 
 
# Define LangGraph Workflow
class ChatState:
    pass  # Define state attributes if needed
 
workflow = StateGraph(ChatState)
workflow.add_node("validate", check_guardrails)
workflow.add_entry_point("validate")  
workflow.add_finish_point("validate")      
graph = workflow.compile()
 
 
# Independent Execution for Testing
if __name__ == "__main__":
    user_message = input("Enter your message: ")
    validation_result = check_guardrails(user_message)
    print(f"\nGuardrails Decision: {validation_result}")
 