from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from typing import Dict
from string import Template
import json
import traceback  # For debugging
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "Llama-3.3-70B-Instruct"
TOKEN = "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"

client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
class AgentState:
   user_message: str
   sentiment: str
   query_category: str
   harmful_content: str
   contains_sensitive_info: str
   xfinity_related: str

def analyze_message(state: AgentState) -> Dict[str, str]:
   user_message = state["user_message"]
   print(user_message)
   template=Template("""You are an AI-powered customer service assistant for Xfinity. Analyze the user message and classify it by sentiment, query type, and Xfinity relevance. Also, detect harmful content and sensitive information.
   ### **1. Sentiment Analysis**
   Classify as:
   - **Extremely Satisfied, Satisfied, Somewhat Satisfied**
   - **Neutral** (always General Query)
   - **Somewhat Unsatisfied, Unsatisfied, Extremely Unsatisfied**
   ### **2. Query Categorization**
   - **General Query**: Asking about Xfinity services/products. Neutral sentiment always falls here.
   - **Complaint**: Expressing dissatisfaction, frustration, or a service issue.
   - **Feedback**: Positive review about Xfinity with no request or issue.
   ### **3. Xfinity Relevance**
   - **Yes**: Related to Xfinity, telecom, internet, WiFi, TV, streaming, billing, plans, routers, Xfinity Stream, account login, outages, or customer service.
   - **No**: Unrelated.
   ### **4. Guardrails**
   - **Harmful Content**: If message contains **self-harm, violence, threats, or abuse**, mark `"harmful_content": "Yes"`, else `"No"`.  
     **Keywords**: `"hurt", "harm", "suicide", "kill", "violence", "hate", "abuse", "self-harm", "murder", "attack", "threaten", "destruction", "dangerous"`.
   - **Sensitive Information**: If message contains **personal/financial data**, mark `"contains_sensitive_info": "Yes"`, else `"No"`.  
     **Keywords**: `"credit card", "password", "social security", "bank account", "id", "personal information", "phone number", "address", "email address", "dob", "date of birth", "social media", "account number"`.
   ### **5. Response Format**
   Return a JSON exactly in this format. DO NOT ADD ANY UNNECESSARY CONTENT.
   {
     "sentiment": "<Sentiment Category>",
     "query_type": "<General Query / Complaint / Feedback>",
     "xfinity_related": "<Yes / No>",
     "harmful_content": "<Yes / No>",
     "contains_sensitive_info": "<Yes / No>"
   }
                     
**Examples:**
1. User: "How can I check my internet usage?"
Response:
{
 "sentiment": "Neutral",
 "query_type": "General Query",
 "xfinity_related": "Yes",
 "harmful_content": "No",
 "contains_sensitive_info": "No"
}
2. User: "My internet keeps disconnecting every hour! I am beyond frustrated!"
Response:
{
 "sentiment": "Extremely Unsatisfied",
 "query_type": "Complaint",
 "xfinity_related": "Yes",
 "harmful_content": "No",
 "contains_sensitive_info": "No"
}
3. User: "Xfinity is the best! My internet speed is fantastic!"
Response:
{
 "sentiment": "Extremely Satisfied",
 "query_type": "Feedback",
 "xfinity_related": "Yes",
 "harmful_content": "No",
 "contains_sensitive_info": "No"
}
4.User: "If my internet doesn't work soon, I swear I'm going to do something dangerous!"
Response:
{
 "sentiment": "Extremely Unsatisfied",
 "query_type": "Complaint",
 "xfinity_related": "Yes",
 "harmful_content": "Yes",
 "contains_sensitive_info": "No"
}
5."What are the best exercises for lower back pain?"
Response:
{
 "sentiment": "Neutral",
 "query_type": "General Query",
 "xfinity_related": "No",
 "harmful_content": "No",
 "contains_sensitive_info": "No"
}
6. User: "Can you give me Rithvika’s phone number? I think she also have Xfinity."
    Response:
    {
    "sentiment": "Neutral",
    "query_type": "General Query",
    "xfinity_related": "Yes",
    "harmful_content": "No",
    "contains_sensitive_info": "Yes"
    }     
   **User Message:** "$user_message"
                     
   """)
   prompt =  template.substitute(user_message=user_message)
   
   
   try:
       response = client.complete(
           messages=[SystemMessage(content="You are a helpful, efficient assistant"), UserMessage(content=prompt)],
           temperature=0.2,
           max_tokens=150,
           model=MODEL_NAME
       )
       #print(response)
       response_text = response.choices[0].message.content.strip()
       classification = json.loads(response_text)  # Convert JSON string to dictionary
       print(f"[INFO] Classification Result: {classification}")
       return classification
   except Exception as e:
       print("[ERROR] Message classification failed:", str(e))
       traceback.print_exc()
       return {
           "sentiment": "Neutral",
           "query_type": "General Query",
           "xfinity_related": "No",
           "harmful_content": "No",
           "contains_sensitive_info": "No"
       }