from typing import Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from agents.message_handling_agent import analyze_message  
from agents.query_handling_agent import handle_query
from agents.complaint_handling_agent import handle_complaint
import traceback  # For debugging
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import json
from string import Template

ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "Llama-3.3-70B-Instruct"
TOKEN = "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"
client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))

class OrchestratorState(TypedDict):
   user_message: str
   sentiment: str
   category: str
   response: str
   buttons: List[Dict[str, str]]
   needs_confirmation: bool
   harmful_content: str
   contains_sensitive_info: str
   xfinity_related: str

# **Step 1: Handle Confirmation Before Classification**
def handle_confirmation(state: OrchestratorState) -> Dict[str, str]:
   print("[INFO] Handling Confirmation...")
   user_response = state["user_message"].strip().lower()
   if "yes" in user_response:
       return {
           **state,
           "response": "I'm so glad that I could help. Please let me know if you have any more queries.",
           "buttons": [],
           "needs_confirmation": False  # ✅ Reset confirmation state
       }
   elif "no" in user_response:
       return {
           **state,
           "response": "I’m sorry that I couldn’t fully clarify your question. Would you like to rephrase it, or I can connect you to an agent?",
           "buttons": [
               {"label": "Chat with Agent", "action": "chat_agent"},
               {"label": "Call the Agent", "action": "call_agent"}
           ],
           "needs_confirmation": False  # ✅ Reset confirmation state
       }
   else:
       return {
           **state,
           "response": "Please enter Yes or No",
           "buttons": [],
           "needs_confirmation": True
       }
# **Step 2: Classify Message (Only Determines Variables, No Routing)**
def classify_message(state: OrchestratorState) -> OrchestratorState:
   try:
       # ✅ **Skip classification if needs_confirmation is already True**
       if state["needs_confirmation"]:
           print("[INFO] Skipping classification, going to confirmation.")
           return state  # ✅ Return state as-is, directly routes to handle_confirmation
       print("[INFO] Inside classification...")
       classification = analyze_message({"user_message": state["user_message"]})
       print(f"[INFO] Classification Result: {classification}")
       return {
           **state,  # ✅ Preserve previous state variables
           "sentiment": classification["sentiment"],
           "category": classification["query_type"],
           "xfinity_related": classification["xfinity_related"],
           "harmful_content": classification["harmful_content"],
           "contains_sensitive_info": classification["contains_sensitive_info"],
           "needs_confirmation": False  # ✅ Default False after classification
       }
   except Exception as e:
       print("[ERROR] Classification failed:", str(e))
       traceback.print_exc()
       return {
           **state,
           "sentiment": "Neutral",
           "category": "General Query",
           "xfinity_related": "No",
           "harmful_content": "No",
           "contains_sensitive_info": "No",
           "needs_confirmation": False
       }
# **Step 3: Process Queries**
def handle_query_agent(state: OrchestratorState) -> Dict[str, str]:
   print("[INFO] Handling General Query...")
   if state["xfinity_related"] == "No":
       return {
           **state,
           "response": "I think this query might be out of context. Please ask a query related to Xfinity.",
           "buttons": [],
           "needs_confirmation": False
       }
   query_result = handle_query({"user_query": state["user_message"]})
   if query_result["needs_confirmation"]:
       return {
           **state,
           "response": query_result["retrieved_response"] + "\n\n Did this clarify your question? (Yes/No)",
           "buttons": [],
           "needs_confirmation": True
       }
   return {
       **state,
       "response": query_result["retrieved_response"],
       "buttons": query_result.get("buttons", []),
       "needs_confirmation": False
   }
# **Step 4: Handle Complaints**
def handle_complaint_agent(state: OrchestratorState) -> Dict[str, str]:
   print("[INFO] Handling Complaint...")
   complaint_result = handle_complaint({
       "user_complaint": state["user_message"],
       "sentiment": state["sentiment"],
       "complaint_response": "",
       "buttons": []
   })
   return {
       **state,
       "response": complaint_result["complaint_response"],
       "buttons": complaint_result["buttons"],
       "needs_confirmation": False
   }
# **Step 5: Handle Feedback**
def handle_feedback(state: OrchestratorState) -> Dict[str, str]:
   print("[INFO] Handling Feedback...")
   return {
       **state,
       "response": "Thank you for your feedback! We really appreciate your input. Please let me know if you have any more queries.",
       "buttons": [],
       "needs_confirmation": False
   }
# **Step 6: Guardrails for Harmful or Sensitive Content**
def guardrails(state: OrchestratorState) -> Dict[str, str]:
   print("[INFO] Checking Guardrails...")
   if state["harmful_content"] == "Yes":
       return {
           **state,
           "response": "We are unable to process this request as it may contain content that isn't safe. Let us know if there's anything else we can assist you with.",
           "buttons": [],
           "needs_confirmation": False
       }
   if state["contains_sensitive_info"] == "Yes":
       return {
           **state,
           "response": "For privacy and security reasons, we are unable to process this request. If there is anything else we can help you with, feel free to ask.",
           "buttons": [],
           "needs_confirmation": False
       }
   return state  # Continue normal flow


def decide_next_step(state: OrchestratorState) -> str:
   """Uses LLM to decide which agent should handle the next step dynamically """
   prompt_template = Template("""
   You are an AI workflow orchestrator managing a customer service chatbot.
   Your task is to **choose the next best step** based on the user's query **following these rules**.
   ---
   **User Message:** "$user_message"
   **category:** "$category"
   **sentiment:** "$sentiment"
   **xfinity_related:** "$xfinity_related"
   **contains_sensitive_info:** "$contains_sensitive_info"
   **harmful_content:** "$harmful_content"
   **needs_confirmation:** "$needs_confirmation"
   ---
   **Decision Criteria (Follow These Rules Exactly):**
   - If `"needs_confirmation"` is `"True"`, select `"handle_confirmation"` (this step should be handled first before anything else).
   - If `"harmful_content"` is `"Yes"` or `"contains_sensitive_info"` is `"Yes"`, select `"guardrails"`.
   - If `"category"` is `"General Query"`, select `"handle_query_agent"`.
   - If `"category"` is `"Complaint"`, select `"handle_complaint_agent"`.
   - If `"category"` is `"Feedback"`, select `"handle_feedback"`.
   **You MUST select one of the following exact action names:**
   - `"handle_confirmation"` (If user needs to confirm something before continuing)
   - `"handle_query_agent"` (For answering general queries)
   - `"handle_complaint_agent"` (For handling complaints)
   - `"handle_feedback"` (For logging feedback)
   - `"guardrails"` (For blocking harmful or sensitive content)
   **Return the next step in the following JSON format (DO NOT include reasoning):**
   {
       "next_step": "<One of: handle_query_agent, handle_complaint_agent, handle_feedback, guardrails, handle_confirmation>"
   }
   

   Examples:
   1)
    Input:
    {
    "user_message": "How do I reset my router?",
    "category": "General Query",
    "sentiment": "Neutral",
    "xfinity_related": "Yes",
    "contains_sensitive_info": "No",
    "harmful_content": "No",
    "needs_confirmation": "False"
    }
    Response:
    {
    "next_step": "handle_query_agent"
    }
    2)Input:
    {
    "user_message": "Yes, that helps!",
    "category": "General Query",
    "sentiment": "Neutral",
    "xfinity_related": "Yes",
    "contains_sensitive_info": "No",
    "harmful_content": "No",
    "needs_confirmation": "True"
    }
    Response:
        {
        "next_step": "handle_query_agent"
        }
    3)Input :
    {
   "user_message": "My internet is too slow, and I am very frustrated!",
   "category": "Complaint",
   "sentiment": "Extremely Unsatisfied",
   "xfinity_related": "Yes",
   "contains_sensitive_info": "No",
   "harmful_content": "No",
   "needs_confirmation": "False"
    }
    Response :
    {
    "next_step": "handle_complaint_agent"
    }
    4)Input:
    {
    "user_message": "Can you help me with my account number?",
    "category": "General Query",
    "sentiment": "Neutral",
    "xfinity_related": "Yes",
    "contains_sensitive_info": "Yes",
    "harmful_content": "No",
    "needs_confirmation": "False"
    }
    Response :
    {
    "next_step": "guardrails"
    }
   """)

   prompt = prompt_template.substitute(
       user_message=state["user_message"],
       category=state["category"],
       sentiment=state["sentiment"],
       xfinity_related=state["xfinity_related"],
       contains_sensitive_info=state["contains_sensitive_info"],
       harmful_content=state["harmful_content"],
       needs_confirmation=state["needs_confirmation"]
   )
   response = client.complete(
       messages=[SystemMessage(content="Decide the best agent to call next"), UserMessage(content=prompt)],
       temperature=0.1,  # Reduce randomness for structured reasoning
       max_tokens=50,  # Lower token usage since reasoning is removed
       model=MODEL_NAME
   )
   decision = json.loads(response.choices[0].message.content.strip())
   print(f"[INFO] LLM Decision: {decision['next_step']}")
   return decision["next_step"]


graph = StateGraph(OrchestratorState)
graph.add_node("classify_message", classify_message)
graph.add_node("handle_query_agent", handle_query_agent)
graph.add_node("handle_complaint_agent", handle_complaint_agent)
graph.add_node("handle_feedback", handle_feedback)
graph.add_node("handle_confirmation", handle_confirmation)
graph.add_node("guardrails", guardrails)
# **Routing Logic**
graph.set_entry_point("classify_message")
# **Conditional Routing**
graph.add_conditional_edges("classify_message", decide_next_step)
# **End points**
graph.set_finish_point("handle_query_agent")
graph.set_finish_point("handle_complaint_agent")
graph.set_finish_point("handle_feedback")
graph.set_finish_point("handle_confirmation")
graph.set_finish_point("guardrails")
orchestrator_agent = graph.compile()

