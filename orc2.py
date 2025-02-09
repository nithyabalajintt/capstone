from typing import Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from agents.message_handling_agent import analyze_message
from agents.query_handling_agent import handle_query
from agents.complaint_handling_agent import handle_complaint
from agents.multi_lingual_agent import detect_language, translate_text
 
class OrchestratorState(TypedDict):
    user_message: str
    sentiment: str
    category: str
    response: str
    buttons: List[Dict[str, str]]
    needs_confirmation: bool
 
def preprocess_message(state: OrchestratorState) -> OrchestratorState:
    detected_language = detect_language(state['user_message'])
    if detected_language != 'en':
        translated_message = translate_text(state['user_message'], 'en')
        state['user_message'] = translated_message
    return state
 
def classify_message(state: OrchestratorState) -> OrchestratorState:
    if state['needs_confirmation']:
        return handle_confirmation(state)
    
    classification = analyze_message({
        "user_message": state["user_message"],
        "sentiment": "",
        "query_category": ""
    })
    state.update({
        "sentiment": classification["sentiment"],
        "category": classification["query_category"],
        "needs_confirmation": False
    })
    return state
 
def route_message(state: OrchestratorState) -> OrchestratorState:
    if state["category"] == "General Query":
        response = handle_query({
            "user_query": state["user_message"],
            "retrieved_response": "",
            "sources": [],
            "needs_confirmation": False,
            "buttons": []
        })
 
        if response["retrieved_response"] in [
            "Sorry, I think this question is out of context. I don't have knowledge on this topic. Please ask a question related to Xfinity.",
            "Sorry, I couldn't find an exact answer for your question in my knowledge base. I can connect you to an agent for further assistance.",
        ]:
            state.update({
                "response": response["retrieved_response"],
                "buttons": response.get("buttons", []),
                "needs_confirmation": False
            })
            return state
        
        if response["needs_confirmation"]:
            final_response = response["retrieved_response"] + "\n\n Did this answer your question? (Yes/No)"
            state.update({
                "response": final_response,
                "buttons": [],
                "needs_confirmation": True
            })
            return state
        
        state.update({
            "response": response["retrieved_response"],
            "buttons": [],
            "needs_confirmation": False
        })
        return state
    
    elif state["category"] == "Complaint":
        response = handle_complaint({
            "user_complaint": state["user_message"],
            "sentiment": state["sentiment"],
            "complaint_response": "",
            "buttons": []
        })
 
        state.update({
            "response": response["complaint_response"],
            "buttons": response["buttons"],
            "needs_confirmation": False
        })
        return state
 
def handle_confirmation(state: OrchestratorState) -> OrchestratorState:
    user_response = state["user_message"].strip().lower()
    if user_response == "yes":
        state.update({
            "response": "I'm glad I could help. Do you have any more queries?",
            "buttons": [],
            "needs_confirmation": False
        })
    elif user_response == "no":
        state.update({
            "response": "I’m sorry that I couldn’t fully clarify your question. Would you like to rephrase it, or I can connect you to an agent?",
            "buttons": [
                {"label": "Chat with Agent", "action": "chat_agent"},
                {"label": "Call the Agent", "action": "call_agent"}
            ],
            "needs_confirmation": False
        })
    else:
        state.update({
            "response": "Please enter Yes or No",
            "buttons": [],
            "needs_confirmation": True
        })
    return state
 
graph = StateGraph(OrchestratorState)
graph.add_node("preprocess_message", preprocess_message)
graph.add_node("classify_message", classify_message)
graph.add_node("route_message", route_message)
graph.add_node("handle_confirmation", handle_confirmation)
graph.set_entry_point("preprocess_message")
graph.add_edge("preprocess_message", "classify_message")
graph.add_edge("classify_message", "route_message")
graph.set_finish_point("route_message")
graph.set_finish_point("handle_confirmation")
orchestrator_agent = graph.compile()
 
 
::contentReference[oaicite:0]{index=0}
 