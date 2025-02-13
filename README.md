# AI RAG with FAISS

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** using **LangChain**, **FAISS**, and **OpenAI**. The dataset (movies.csv) is converted into chunks and stored in a **vector database**. When a user queries, the system retrieves the top 5 relevant movies and generates a response using OpenAI's language model.

## Tech Stack
- **Backend**: Flask (REST API for handling queries)
- **Frontend**: Streamlit (UI for user interaction)
- **Vector Database**: FAISS (to store and retrieve embeddings)
- **Embeddings & LLM**: OpenAI (to generate and process responses)
- **Data Processing**: Pandas (for handling the dataset)

## Installation and Setup
### 1. Clone the Repository
```sh
git clone <repo-url>
cd <repo-name>
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Set Up OpenAI API Key
Store your **OpenAI API Key** in OPENAI_API_Key.txt:


### 4. Run the Flask Backend
```sh
python app.py
```
This will start the Flask API on **http://127.0.0.1:5000**

### 5. Run the Streamlit Frontend
Open another terminal and run:
```sh
streamlit run streamlit_app.py
```
This will launch the **Streamlit UI** where users can interact with the AI assistant.

## API Endpoint
### `POST /chat`
**Request:**
```json
{
  "query": "Tell me about sci-fi movies"
}
```
**Response:**
```json
{
  "response": "Based on the following movies...",
  "history": [
    ["User Query 1", "Response 1"],
    ["User Query 2", "Response 2"]
  ]
}
```

## Features
- **Stores & retrieves embeddings using FAISS**
- **Retrieves top 5 relevant movie descriptions**
- **Persistent chat history (last 5 messages)**
- **User-friendly Streamlit UI**

