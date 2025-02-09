from typing import Dict, List
from typing_extensions import TypedDict
import traceback  # For debugging
class ComplaintAgentState(TypedDict):
  user_complaint: str
  sentiment: str
  complaint_response: str
  buttons: List[Dict[str, str]]  
# Complaint response templates based on updated sentiment categories
complaint_template = {
  "extremely unsatisfied": "We sincerely apologize for the inconvenience. Your issue has been escalated, and a ticket has been raised. A representative will reach out as soon as possible. If you need immediate assistance, please select '💬 Chat with Agent' or '📞 Call the Agent'.",
  "unsatisfied": "We understand your frustration and appreciate your patience. A complaint ticket has been raised, and our support team will contact you shortly. If you need urgent assistance, please use the options below.",
  "somewhat unsatisfied": "We're sorry that you're experiencing issues. Your concern has been recorded, and our team will investigate the matter. If you require additional support, please select one of the options below.",
  "neutral": "Your complaint has been registered, and we will provide updates soon. If you need further assistance, please feel free to reach out using the options below.",
  "satisfied": "Thank you for your feedback! If there's anything else we can assist with, please let us know. You can also reach out using the buttons below.",
  "somewhat satisfied": "We're glad to hear that things are improving. If you need any more help, please don't hesitate to reach out using the available options.",
  "extremely satisfied": "Thank you for your kind words! If there's anything else we can do for you, let us know.",
  "default": "Thank you for reaching out. Your complaint has been recorded, and we will ensure it is addressed promptly. For urgent support, please select '💬 Chat with Agent' or '📞 Call the Agent'."
}
def handle_complaint(state: ComplaintAgentState) -> Dict[str, str]:
   """
   Handles customer complaints by generating appropriate responses based on sentiment.
   """
   try:
       sentiment = state["sentiment"].lower()
       # Select appropriate response based on sentiment
       complaint_response = complaint_template.get(sentiment, complaint_template["default"])
       # Provide buttons for further assistance in ALL cases
       buttons = [
           {"label": "Chat with Agent", "action": "chat_agent"},
           {"label": "Call the Agent", "action": "call_agent"}
       ]
       return {"complaint_response": complaint_response, "buttons": buttons}
   
   except Exception as e:
       print("[ERROR] Complaint handling failed:", str(e))
       traceback.print_exc()
       return {
           "complaint_response": "We're sorry, but we couldn't process your complaint at this moment. Please try again later.",
           "buttons": [
               {"label": "Chat with Agent", "action": "chat_agent"},
               {"label": "Call the Agent", "action": "call_agent"}
           ]
       }