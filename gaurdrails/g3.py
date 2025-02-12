import json
import yaml
import faiss
import numpy as np
from typing import Dict
from types import SimpleNamespace
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from nemoguardrails import LLMRails
from langgraph.graph import StateGraph, END
 
# Azure AI Model Configuration (Using GPT-Mini-4o)
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-mini-4o"
TOKEN = "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"
 
# Initialize Azure Clients
chat_client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
embedding_client = EmbeddingsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
 
#loading configuration
 
class DotDict(dict):
    """Allows accessing dictionary keys as attributes, with defaults."""
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            if key == "flows":
                return []  # ✅ Return an empty list if 'flows' is missing
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
 
    def __setattr__(self, key, value):
        self[key] = value
 
    @classmethod
    def from_dict(cls, obj):
        """Recursively convert a dictionary to DotDict"""
        if isinstance(obj, dict):
            return cls({k: cls.from_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [cls.from_dict(i) for i in obj]
        return obj
 
# Load YAML as dictionary
with open("config.yml", "r") as f:
    config_dict = yaml.safe_load(f)
 
# Convert dictionary to DotDict
config_obj = DotDict.from_dict(config_dict)
 
# ✅ Ensure 'flows' is always present
if "flows" not in config_obj:
    config_obj.flows = []
 
# ✅ Print to confirm it's working
print(config_obj.get("colang_version", "No colang_version found"))  # Should print the version
print(config_obj.flows)  # Should print an empty list if missing
 
# ✅ Pass to LLMRails
rails = LLMRails(config=config_obj)
    

# Initialize LangGraph
graph = StateGraph()
 
# Function to get text embeddings
def get_embedding(text: str):
    """Generates an embedding vector for the given text using Azure AI."""
    response = embedding_client.embed(model=MODEL_NAME, input=[text])
    return response.embeddings[0]  # Assuming single text input
 
# Detecting embedding size dynamically
sample_text = "This is a test sentence."
sample_embedding = get_embedding(sample_text)
dimension = len(sample_embedding)
 
# Create FAISS index
index = faiss.IndexFlatL2(dimension)
 
# Function to detect language
def detect_language(text: str) -> str:
    """Detects if the input language is English or Spanish."""
    lang = rails.detect_language(text)
    if lang not in ["en", "es"]:
        return "unsupported"
    return lang
 
# Guardrails validation function
def check_guardrails(state: Dict) -> Dict:
    """Validates user input using different checks: harmful, sensitive, profanity, spam, very short, and language."""
    user_input = state.get("user_input", "")
    print("[INFO] Running Guardrails validation...")
 
    # 1. Language Detection
    lang = detect_language(user_input)
    if lang == "unsupported":
        return {"status": "error", "reason": "language_not_supported", "message": "We currently support only English and Spanish."}
 
    # 2. Process input with Azure AI Model
    messages = [SystemMessage(content="Validate this user input"), UserMessage(content=user_input)]
    response = chat_client.complete(model=MODEL_NAME, messages=messages)
    content = response.choices[0].message.content.strip().lower()
 
    # 3. Guardrails-based filtering
    if "harmful" in content:
        return {"status": "error", "reason": "harmful_content", "message": "This request contains harmful content."}
    elif "sensitive" in content:
        return {"status": "error", "reason": "sensitive_content", "message": "This request contains sensitive information."}
    elif "profanity" in content:
        return {"status": "error", "reason": "profanity", "message": "Please avoid using inappropriate language."}
    elif "spam" in content:
        return {"status": "error", "reason": "spam_detected", "message": "Your message appears to be spam."}
    elif len(user_input.strip()) < 5:
        return {"status": "error", "reason": "short_message", "message": "Your message is too short. Please provide more details."}
 
    # 4. FAISS similarity check (Example: comparing input to stored messages)
    query_embedding = get_embedding(user_input)
    query_embedding = np.array([query_embedding], dtype=np.float32)  # Convert to FAISS format
 
    if index.ntotal > 0:  # Check similarity only if there are stored embeddings
        D, I = index.search(query_embedding, k=1)  # Find closest match
        if D[0][0] < 0.5:  # Threshold for similarity (adjustable)
            return {"status": "error", "reason": "similar_content_detected", "message": "Your message is too similar to existing content."}
 
    return {"status": "success", "reason": "validated", "message": "Your request is valid and being processed."}
 
# Adding to LangGraph
graph.add_node("validate_input", check_guardrails)
graph.add_entry_point("validate_input")
graph.add_finish_point("validate_input")
 
# Independent Execution for Testing
if __name__ == "__main__":
    user_message = input("Enter your message: ")
    state = {"user_input": user_message}
    validation_result = graph.invoke(state)
    print(f"\nGuardrails Decision: {validation_result}")
 