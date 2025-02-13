from flask import app
import requests
import streamlit as st

# Streamlit UI
st.title("AI RAG Movie Assistant")
user_input = st.text_input("Ask a movie-related question:")

if st.button("Ask"): 
    response = requests.post("http://127.0.0.1:5000/chat", json={"query": user_input})
    st.write("Response:", response.json()["response"])
    
    st.write("Chat History:")
    for q, r in response.json()["history"]:
        st.text(f"Q: {q}\nA: {r}")

