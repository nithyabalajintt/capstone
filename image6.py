import easyocr  # OCR for text extraction
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from langgraph.graph import StateGraph
from typing import Dict

# Load CLIP and BLIP models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize EasyOCR reader for OCR text extraction
ocr_reader = easyocr.Reader(['en'])  

AgentState = Dict[str, str]

# **1. Extract ENTIRE Text from First Image (Xfinity Bill)**
def extract_full_text_from_bill(image_path: str) -> str:
    """Extracts all text, numbers, and symbols from the Xfinity bill."""
    results = ocr_reader.readtext(image_path)
    full_text = " ".join([text[1] for text in results])
    return full_text.strip()

# **2. Generate a Detailed Caption for Second Image**
def generate_detailed_caption(image_path: str) -> str:
    """Generates a detailed description of what is happening in the image."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption  

# **3. Identify What is Happening in the Second Image**
def analyze_image_content(image_path: str) -> str:
    """Uses CLIP to understand what is happening in the image."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    image_embedding = clip_model.get_image_features(**inputs)

    # Possible scenarios
    text_descriptions = [
        "A father and child playing with toys",
        "A single child playing with blocks", "A man using a laptop",
        "A couple watching TV", "A person using a mobile phone",
        "A family enjoying time together", "A person fixing a cable connection"
    ]

    text_inputs = clip_processor(text=text_descriptions, padding=True, return_tensors="pt")
    text_embeddings = clip_model.get_text_features(**text_inputs)

    # Find the best match
    similarity = torch.cosine_similarity(image_embedding, text_embeddings, dim=-1)
    best_match_index = similarity.argmax().item()

    return text_descriptions[best_match_index]

# **4. Main Image Processing Function**
def analyze_images(state: AgentState) -> AgentState:
    bill_image_path = r"C:\Users\338565\venv\chatbot\Comcast Bill Sample.jpg"  # First image (Bill)
    second_image_path = r"C:\Users\338565\venv\chatbot\image2.jpg"  # Second image (Activity)

    # **Extract Full Text from Bill**
    extracted_text = extract_full_text_from_bill(bill_image_path)

    # **Generate Description for Second Image**
    blip_caption = generate_detailed_caption(second_image_path)
    image_content_description = analyze_image_content(second_image_path)

    # Store results in state
    state["full_text_bill"] = extracted_text
    state["second_image_description"] = image_content_description  
    state["blip_caption"] = blip_caption  

    return state

# **5. LangGraph Workflow**
graph = StateGraph(AgentState)
graph.add_node("image_analysis", analyze_images)
graph.set_entry_point("image_analysis")
graph.set_finish_point("image_analysis")
image_analysis_agent = graph.compile()

# **6. Run the Code**
def main():
    print("Starting the Image Processing Agent")
    result = image_analysis_agent.invoke({})

    print("\n✅ **Extracted Text from Xfinity Bill:**")
    print(result.get("full_text_bill", "No text found"))

    print("\n✅ **AI-Based Image Content Understanding (Second Image):**")
    print(f"CLIP Analysis: {result.get('second_image_description', 'N/A')}")
    print(f"BLIP Captioning: {result.get('blip_caption', 'N/A')}")

if __name__ == "__main__":
    main()

