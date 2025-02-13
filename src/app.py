import os
import pandas as pd
import openai
from flask import Flask
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings,ChatOpenAI
from flask import jsonify, request



# Load movies dataset
movies = pd.read_csv("movies.csv")

# Fill NaN values in 'overview' column with an empty string
movies["overview"] = movies["overview"].fillna("")


# Initialize OpenAI API Key
with open("OpenAI_API_Key.txt", "r") as f:
  openai.api_key = ' '.join(f.readlines())

client = openai.OpenAI(api_key=openai.api_key)
# Process dataset into chunks
def chunk_data(data):
    return [Document(page_content=row["overview"], metadata={"id": row["id"], "title": row["title"], "release_date": row["release_date"], "popularity": row["popularity"], "vote_count": row["vote_count"], "vote_average": row["vote_average"]}) for _, row in data.iterrows()]

# Set up the file path for the FAISS index
faiss_index_path = "faiss_index"

# Create FAISS index and ingesting the movies data as embeddings 
embeddings = OpenAIEmbeddings(api_key=openai.api_key, model = "text-embedding-ada-002")

if os.path.exists(faiss_index_path):
    
    # Load the existing embeddings from the FAISS index
    print("FAISS index already exists. Loading existing index...")
    vector_db = FAISS.load_local(faiss_index_path, embeddings,allow_dangerous_deserialization=True)
else:
    docs = chunk_data(movies)
    # Ensure 'docs' has no empty or invalid data
    docs = [doc for doc in docs if doc.strip()]
    if not docs:
        raise ValueError("No valid documents to embed.")

    # Check and preprocess docs before passing to FAISS
    try:
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(faiss_index_path)
        print("FAISS index created and saved locally.")
    except Exception as e:
        print(f"Error during FAISS processing: {e}")
        raise


# Initialize Flask app
app = Flask(__name__)

chat_model = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai.api_key)
chat_history = []



# Create /chat endpoint to get llm response.
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    user_query = request.json.get("query")
    
    # Retrieve top 5 similar results
    def retrieve_context(user_query, top_k=5):
        """Retrieve the top K most relevant documents from FAISS"""
        results = vector_db.similarity_search(user_query, k=top_k)
        
        # Format the context
        context = "\n".join([
            f"Title: {doc.metadata['title']}, Release Date: {doc.metadata['release_date']},Popularity: {doc.metadata['popularity']},Vote count: {doc.metadata['vote_count']}\nOverview: {doc.page_content}"
            for doc in results
        ])
        
        return context

    def get_response_from_llm(context, user_query):
        system_message = {
            "role": "system",
            "content": f"Answer solely based on the following context:\n<Documents>\n{context}\n</Documents>",
        }
        user_message = {
            "role": "user",
            "content": user_query,
        }
        messages = [system_message, user_message]

        # Send to OpenAI ChatCompletion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages
        )
        return response.choices[0].message.content

    context = retrieve_context(user_query)
    response = get_response_from_llm(context, user_query)
    print(response)

    # Maintain last 5 interactions
    chat_history.append((user_query, response))
    if len(chat_history) > 3:
        chat_history.pop(0)
    
    return jsonify({"response": response, "history": chat_history})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')