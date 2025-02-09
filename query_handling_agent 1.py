import os
from typing import Dict
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import traceback  # For debugging

import certifi
import ssl
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

ssl._create_default_https_context=ssl._create_unverified_context

class QueryAgentState(TypedDict):
   user_query: str
   retrieved_response: str
   sources: str
   needs_confirmation: bool

# Azure AI Config
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "Llama-3.3-70B-Instruct"
TOKEN = "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"
client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))

# FAISS Vector Database Config
FAISS_INDEX_PATH = "faiss index"
AZURE_OPENAI_API_KEY = "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"

def load_vector_store():
   if not os.path.exists(FAISS_INDEX_PATH):
       raise ValueError("[ERROR] FAISS index not found! Run `setup_KB.py` first.")
   print("[INFO] Loading FAISS vector store...")
   return FAISS.load_local(
       FAISS_INDEX_PATH,
       OpenAIEmbeddings(
           model="text-embedding-3-small",
           openai_api_key=AZURE_OPENAI_API_KEY,
           openai_api_base=ENDPOINT
       ),
       allow_dangerous_deserialization=True
   )

vector_store = load_vector_store()

def retrieve_context(query, vector_store, top_k=3):
   try:
       docs = vector_store.similarity_search(query, k=top_k)
       if not docs:
           return None, []
       results = []
       sources = []
       for doc in docs:
           source = doc.metadata.get("source", "Unknown Source")
           results.append(f"**Content:** {doc.page_content}")
           sources.append(source)
       return "\n\n".join(results), sources
   except Exception as e:
       print("[ERROR] Context retrieval failed:", str(e))
       traceback.print_exc()
       return None, []
   
def query_gpt4o(user_query, context):

   try:
       prompt = f"""Use the following retrieved context to answer the query **without explicitly referencing it**.
       **Context:**
       {context}
       **Query:**
       {user_query}
       **Provide a direct and natural response as if you already know the information. DO NOT say phrases like 'According to the information provided' or 'Based on the retrieved context'. Just give the answer.**
       """
       response = client.complete(
           messages=[
               SystemMessage(content="You are a knowledgeable customer service assistant."),
               UserMessage(content=prompt)
           ],
           temperature=0.5,
           max_tokens=500,
           model=MODEL_NAME
       )
       return response.choices[0].message.content.strip()
   except Exception as e:
       print("[ERROR] GPT-4o query failed:", str(e))
       traceback.print_exc()
       return "Sorry, something went wrong while generating a response."
def handle_query(state: QueryAgentState) -> Dict[str, str]:

   user_query = state["user_query"]
   print(f"[INFO] Processing Query: {user_query}")

   if not user_query:
       return {"retrieved_response": "Error: Query cannot be empty.", "sources": [], "needs_confirmation": False}
   # Retrieve context from FAISS

   context, sources = retrieve_context(user_query, vector_store)
   if not context:
       print("[INFO] No relevant context found. Offering agent connection.")
       return {
           "retrieved_response": "Sorry, I couldn't find an exact answer for your question in my knowledge base. I can connect you to an agent for further assistance.",
           "sources": [],
           "needs_confirmation": False,
           "buttons": [
               {"label": "Chat with Agent", "action": "chat_agent"},
               {"label": "Call the Agent", "action": "call_agent"}
           ]
       }
   # Generate response using retrieved context
   retrieved_response = query_gpt4o(user_query, context)
   return {
       "retrieved_response": retrieved_response,
       "sources": sources,
       "needs_confirmation": True  # Ask for confirmation in UI
   }