# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:57:32 2025

run with: streamlit run rag_agent_v1.py 
"""

# =============================================================================
# Imports
# =============================================================================
import streamlit as st
import os
from pathlib import Path
import time
import json
import tiktoken

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Import voice functions from voice_utils.py
from voice_utils import record_and_transcribe, speak_text

st.set_page_config(page_title="Ask H[ai]di")

# Unkomment when pushing... 
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# =============================================================================
# Custom CSS for Small Button
# =============================================================================
st.markdown(
    """
    <style>
    div.stButton > button {
        padding: 0.25em 0.5em;
        font-size: 0.8em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# Paths and Variables
# =============================================================================
project_root = Path(os.getcwd())
CHUNK_SIZE = 800
OVERLAP = 200
MAX_TOKENS = 4096
RESPONSE_BUFFER = 500

FAISS_STORAGE_PATH = project_root / "data" / "faiss_index_800_200"
METADATA_STORAGE_PATH = project_root / "data" / f"metadata_{CHUNK_SIZE}_{OVERLAP}.json"
IMAGE_PATH = project_root / "data" / "heidi_1.png"
GIF_PATH = project_root / "data" / "new_animation.gif"

# =============================================================================
# Helper Functions
# =============================================================================
def calculate_token_length(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

system_prompt = """
Du bist ein Concierge in einem Hotel, der Gästen Auskunft über Restaurants 
in der Umgebung gibt. Du kennst die Gastronomie in der Gegend wie Deine 
Westentasche - und Du bist sehr freundlich und versiert darin, 
Deinen Kunden genau das Richtige zu empfehlen. 

Wichtige Regeln:
- Deine Empfehlung beginnt mit einer einzigen freundlichen Begrüßung wie:
  "Natürlich, ich empfehle Ihnen gerne ein passendes Restaurant..."
  Diese Begrüßung darf **nur am Anfang der gesamten Antwort stehen**, 
  nicht vor jeder einzelnen Empfehlung.
- Du empfiehlst maximal zwei Restaurants und du nennst die Restaurants, die 
  Du empfiehlst, explizit.
- Du gibst Deine Empfehlung in einem Fließtext mit vollständigen Sätzen.  
  Das aus Deiner Sicht beste Restaurant kommt zuerst.
- Falls es keine passende Empfehlung gibt, sag dem Gast das **direkt**, 
  **ohne eine Begrüßung erneut zu wiederholen**.
- Deine Antworten beziehen sich **ausschließlich** auf Restaurants, 
  von denen Dir Dokumente oder Speisekarten vorliegen.
"""

def load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH):
    """
    Loads the FAISS knowledge base and stores it in session state.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.load_local(FAISS_STORAGE_PATH, embeddings, allow_dangerous_deserialization=True)
    st.session_state["knowledge_base"] = knowledge_base
    st.success("H[ai]di ist bereit!")

def generate_response(user_question):
    """Retrieves relevant context and generates an AI response."""
    knowledge_base = st.session_state.get("knowledge_base", None)
    if not knowledge_base:
        return "Kein PDF geladen. Bitte lade zuerst eine Datei hoch."
    
    docs = knowledge_base.similarity_search(user_question)
    context, token_count = "", calculate_token_length(system_prompt + user_question, model_name="gpt-4")
    for doc in docs:
        doc_tokens = calculate_token_length(doc.page_content, model_name="gpt-4")
        if token_count + doc_tokens + RESPONSE_BUFFER < MAX_TOKENS:
            context += doc.page_content + "\n\n"
            token_count += doc_tokens
        else:
            break

    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template("Kontext:\n{context}\n\nFrage: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return qa_chain.run(context=context.strip(), question=user_question)

# =============================================================================
# Main App
# =============================================================================
def main():
    #st.set_page_config(page_title="Ask H[ai]di")
    st.header("Ask H[ai]di")
    
    # Load the knowledge base if not already loaded
    if "knowledge_base" not in st.session_state:
        load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH)
    
    # Placeholder for the image (static or animated)
    image_placeholder = st.empty()
    image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)
    
# Text input for the question
user_question_text = st.text_area("Frage eingeben:")

# Voice input section below the text input
transcript = record_and_transcribe(OPENAI_API_KEY)
if transcript:
    st.session_state["user_question"] = transcript
    st.success("Transkription: " + transcript)
    
    # Use recorded transcript if available; otherwise, use text input.
    user_question = st.session_state.get("user_question", "") or user_question_text
    if st.button("Antwort generieren") and user_question:
        with st.spinner("H[ai]di überlegt..."):
            # Show animated GIF while processing
            image_placeholder.image(GIF_PATH, caption="H[ai]di überlegt...", use_container_width=False)
            time.sleep(3)  # Simulate delay; replace with actual processing time if needed
            response = generate_response(user_question)
            image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)
            st.write(response)
            # Small button to speak the response
            if st.button("Speak it!", key="speak_button"):
                speak_text(response)

if __name__ == "__main__":
    main()
