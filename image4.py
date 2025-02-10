import easyocr  # OCR library for text extraction
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import re
from langgraph.graph import StateGraph
from typing import Dict

# Load CLIP model and processor
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Load BLIP model for image captioning (improves description accuracy)
blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'])  

AgentState = Dict[str, str]

def extract_text_from_bill(image_path: str) -> str:
    """Extracts text from the image using EasyOCR."""
    result = ocr_reader.readtext(image_path)
    extracted_text = " ".join([text[1] for text in result])  
    return extracted_text.strip()

def generate_image_caption(image_path: str) -> str:
    """Uses BLIP to generate a detailed image caption for better description accuracy."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption  

def analyze_image_content(image_path: str) -> str:
    """Uses CLIP to understand the content of an image and match it to predefined Xfinity-related subjects."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    outputs = clip_model.get_image_features(**inputs)
    image_embedding = outputs  

    # Define text descriptions related to Xfinity chatbot topics
    text_descriptions = [
        "A man and a child playing with toys",  # Updated for better detection
        "A single child playing with toys", "A person holding a mobile device",
        "A couple watching TV", "A person using a laptop", "An office workspace",
        "A WiFi router setup", "A customer support representative", "A family watching TV together",
        "A person fixing a cable", "A person browsing an internet bill"
    ]

    text_inputs = clip_processor(text=text_descriptions, padding=True, return_tensors="pt")
    text_embeddings = clip_model.get_text_features(**text_inputs)

    similarity = torch.cosine_similarity(image_embedding, text_embeddings, dim=-1)
    best_match_index = similarity.argmax().item()

    return text_descriptions[best_match_index]  

def analyze_bill(state: AgentState) -> AgentState:
    image_path = r"C:\Users\338565\venv\chatbot\image2.jpg"  # Replace with your actual file path
    extracted_text = extract_text_from_bill(image_path)
   
    # Generate a detailed image description using BLIP
    blip_caption = generate_image_caption(image_path)

    # Analyze the content using CLIP for category matching
    image_content_description = analyze_image_content(image_path)

    # Regular expressions to extract key bill details
    date_pattern = r"\b\d{1,2}/\d{1,2}/\d{4}\b"  
    amount_pattern = r"Total\s+\$?(\d+\.\d{2})"  
    invoice_pattern = r"Invoice\s+#?(\d+)"        

    date_match = re.search(date_pattern, extracted_text)
    amount_match = re.search(amount_pattern, extracted_text)
    invoice_match = re.search(invoice_pattern, extracted_text)

    # Store extracted details in state
    state["invoice_number"] = invoice_match.group(1) if invoice_match else "Not Found"
    state["date"] = date_match.group() if date_match else "Not Found"
    state["total_amount"] = amount_match.group(1) if amount_match else "Not Found"
    state["full_text"] = extracted_text
    state["image_content"] = image_content_description  
    state["blip_caption"] = blip_caption  # Store BLIP-generated caption

    return state

# Create LangGraph workflow
graph = StateGraph(AgentState)
graph.add_node("bill_analysis", analyze_bill)
graph.set_entry_point("bill_analysis")
graph.set_finish_point("bill_analysis")
bill_analysis_agent = graph.compile()

def main():
    print("Starting the Bill Analysis Agent")
    result = bill_analysis_agent.invoke({})
   
    print("\nExtracted Bill Details:")
    print(f"Invoice Number: {result.get('invoice_number', 'N/A')}")
    print(f"Date: {result.get('date', 'N/A')}")
    print(f"Total Amount: {result.get('total_amount', 'N/A')}")
    print("\nFull Extracted Text:\n", result.get("full_text", ""))

    print("\nAI-Based Image Content Understanding:")
    print(f"CLIP Analysis: {result.get('image_content', 'N/A')}")
    print(f"BLIP Captioning: {result.get('blip_caption', 'N/A')}")

if __name__ == "__main__":
    main()
