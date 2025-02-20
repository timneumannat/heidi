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


# =============================================================================
# Login Page
# =============================================================================
def login_page():
    # Always reset the login state on a new run (i.e. on refresh)
    st.session_state["logged_in"] = False

    st.title("Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == st.secrets["password"]["password"]:
            st.session_state["logged_in"] = True
        else:
            st.error("Incorrect password.")

    if not st.session_state["logged_in"]:
        st.stop()



# =============================================================================
# Imports and Global Variables
# =============================================================================
# Unkomment when pushing... 
OPENAI_API_KEY = st.secrets["openai"]["api_key"]


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
    login_page()
    # st.set_page_config(page_title="Ask H[ai]di")
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

    # Clear previous response if a new question is entered.
    if "last_question" not in st.session_state or st.session_state["last_question"] != user_question:
        st.session_state["last_question"] = user_question
        if "response" in st.session_state:
            del st.session_state["response"]
    
    if st.button("Antwort generieren") and user_question:
        with st.spinner("H[ai]di überlegt..."):
            # Show the animated GIF while waiting for the response
            image_placeholder.image(GIF_PATH, caption="H[ai]di überlegt...", use_container_width=False)
    
            # Simulate delay for response generation (replace with actual processing time)
            time.sleep(3)
    
            # Generate the response after the waiting time
            response = generate_response(user_question)
    
            # Show the static image again after the animation
            image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)
    
            # Save the response to session state so it persists outside the block
            st.session_state["response"] = response
            
    # Always display the response if it's stored in session state.
    if "response" in st.session_state:
        st.write(st.session_state["response"])
    
    # Separate "Speak it!" button outside the response generation block.
    if "response" in st.session_state and st.button("Vorlesen", key="speak_button"):
        speak_text(st.session_state["response"])



if __name__ == "__main__":
    main()
