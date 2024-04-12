from flask import Flask, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import mysql.connector
from mysql.connector import Error
from transformers import AutoTokenizer, BartForConditionalGeneration


app = Flask(__name__)

def generate_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def save_embeddings(texts):
    pinecone_client = Pinecone(api_key="e15f4903-51b8-4ed9-9489-cf7883edb8ab")
    pinecone_index = pinecone_client.Index("demo")

    try:
        connection = mysql.connector.connect(
            host='localhost',  
            user='root',
            password='Clicflyer@123',
            database='policy'
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            for i, text in enumerate(texts):
               
                embedding_list = generate_embeddings([text])[0].tolist()  
                document_id = str(i)
                pinecone_index.upsert(vectors=[(document_id, embedding_list)])
                
                
                insert_query = "INSERT INTO documents (document_id, text) VALUES (%s, %s)"
                cursor.execute(insert_query, (document_id, text))
            
            connection.commit()

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    

def fetch_texts_for_ids(db_connection, ids):
    
    query = "SELECT document_id, text FROM documents WHERE document_id IN (%s)"
    
    format_strings = ','.join(['%s'] * len(ids))
    query = query % format_strings
    
    cursor = db_connection.cursor()
    cursor.execute(query, tuple(ids))  
    result = cursor.fetchall()  
    
    id_to_text = {document_id: text for document_id, text in result}
    return id_to_text

def chunk_text(text, max_length):
    words = text.split()
    return [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def summarize_chunks(chunks):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    summaries = []
    for chunk in chunks:
        inputs = tokenizer([chunk], return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=2, max_length=1024, min_length=5)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        summaries.append(summary)
    return summaries






@app.route('/load/<path:pdf_directory>')
def load_files(pdf_directory):
    files = os.listdir(pdf_directory)
    
    pdf_files = [file for file in files if file.endswith('.pdf')]
    all_pages = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
        all_pages.extend(pages)  

    context = "\n".join(str(p.page_content) for p in all_pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=400)
    texts = text_splitter.split_text(context)
    save_embeddings(texts)
    
    #save copy 
    return  jsonify({'Response': "loaded"})



@app.route('/response_query/<query_text>')
def response_query(query_text):
    pinecone_client = Pinecone(api_key="e15f4903-51b8-4ed9-9489-cf7883edb8ab")
    pinecone_index = pinecone_client.Index("demo")
    query_embedding = generate_embeddings([query_text])[0].tolist()
    query_results = pinecone_index.query(vector=query_embedding, top_k=2)
    ids_to_lookup = [match['id'] for match in query_results["matches"]]
    try:
        connection = mysql.connector.connect(
        host='localhost',  
        user='root', 
        password='Clicflyer@123', 
        database='Policy'  
    )
    
        if connection.is_connected():
            id_to_text = fetch_texts_for_ids(connection, ids_to_lookup)

            context = "\n".join(
            id_to_text.get(id, " ")
            for id in ids_to_lookup  # Use the same list of IDs you queried
            )

            input_text = f"question: {query_text} context: {context}"
            # print(input_text)
            chunks = chunk_text(input_text, 1024)  
            chunk_summaries = summarize_chunks(chunks)
            return jsonify({"Response":chunk_summaries})

    finally:
        
        if connection.is_connected():
            connection.close()


if __name__ == '__main__':
    app.run(debug=True)


