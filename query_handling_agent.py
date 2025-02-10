import os
import traceback  # For debugging
import chromadb
from typing import Dict
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


class QueryAgentState(TypedDict):
  user_query: str
  retrieved_response: str
  sources: str
  needs_confirmation: bool
# ✅ Azure AI Config
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"
TOKEN = "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"
client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
# ✅ ChromaDB Config
CHROMA_DB_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(name="knowledge_base")
# ✅ Embedding Model (MiniLM)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# ✅ Sentence Splitter Configuration (For Context Processing)
splitter = RecursiveCharacterTextSplitter(
   chunk_size=300,
   chunk_overlap=50,
   separators=["\n", ". ", "? ", "! "],  # Splits at sentences
   length_function=len
)
def retrieve_context(query, collection, embedding_model, top_k=3):
  """Retrieves top-k most relevant chunks from ChromaDB."""
  try:
      print("🔍 Generating embedding for query...")
      query_embedding = embedding_model.embed_query(query)
      print(f"🔎 Retrieving top {top_k} results from ChromaDB...")
      results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
      if not results["documents"]:
          return None, []
      retrieved_docs = []
      sources = []
      for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
          retrieved_docs.append(doc)
          sources.append(metadata.get("source", "Unknown Source"))
      # ✅ Apply Sentence Splitter to Retrieved Docs (Optional for Better Context)
      processed_docs = []
      for doc in retrieved_docs:
          processed_docs.extend(splitter.split_text(doc))
      return "\n\n".join(processed_docs), sources
  
  except Exception as e:
      print("[ERROR] Context retrieval failed:", str(e))
      traceback.print_exc()
      return None, []
def llm_response(user_query, context):
  """Queries GPT-4o with the retrieved context."""
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
  """Handles incoming user query, retrieves context, and queries GPT-4o."""
  user_query = state["user_query"]
  print(f"[INFO] Processing Query: {user_query}")
  if not user_query:
      return {"retrieved_response": "Error: Query cannot be empty.", "sources": [], "needs_confirmation": False}
  # ✅ Retrieve context from ChromaDB
  context, sources = retrieve_context(user_query, collection, embedding_model)
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
  # ✅ Generate response using retrieved context
  retrieved_response = llm_response(user_query, context)
  return {
      "retrieved_response": retrieved_response,
      "sources": sources,
      "needs_confirmation": True  # Ask for confirmation in UI
  }
