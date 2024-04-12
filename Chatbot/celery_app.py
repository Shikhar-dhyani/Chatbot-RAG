from flask import Flask, jsonify, request
from celery import Celery
from werkzeug.utils import secure_filename
import os
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
from mysql.connector import Error
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone


app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['CELERY_BROKER_URL'] = 'amqp://localhost'  # RabbitMQ as broker

# Initialize Celery with RabbitMQ
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    
    if not files:
        return jsonify(error="No files part"), 400
    
    filenames = []
    skipped_files = []
    for file in files:
        if file.filename == '':
            continue  # Skip empty files
        if not allowed_file(file.filename):
            return jsonify(error=f"File {file.filename} type not allowed"), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check for duplicate file name
        if os.path.exists(save_path):
            skipped_files.append(filename)
            continue  # Skip saving this file
        
        file.save(save_path)
        filenames.append(filename)
        process_pdf_and_generate_embeddings.delay(filename, save_path)
    
    if not filenames and not skipped_files:
        return jsonify(error="No valid files selected"), 400
    message = "Files uploaded successfully" if filenames else "No new files uploaded"
    if skipped_files:
        message += f". Skipped duplicates: {', '.join(skipped_files)}"
    return jsonify(message=message, uploaded_files=filenames, skipped_files=skipped_files), 200

@celery.task
def process_pdf_and_generate_embeddings(filename, save_path):
    pinecone_client = Pinecone(api_key="e15f4903-51b8-4ed9-9489-cf7883edb8ab")
    pinecone_index = pinecone_client.Index("demo")
    pdf_loader = PyPDFLoader(save_path)
    pages = pdf_loader.load_and_split()
    document_text = "\n".join([page.page_content for page in pages])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=400)
    texts = text_splitter.split_text(document_text)
    
    base_document_id = os.path.splitext(filename)[0]
    
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',  
            user='root',
            password='Clicflyer@123',
            database='policy'
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            for i, text_chunk in enumerate(texts, start=1):
                chunk_embedding = generate_embeddings([text_chunk]).tolist()
                document_chunk_id = f"{base_document_id}_chunk_{i}"
                
                # Upsert embeddings into Pinecone
                pinecone_index.upsert(vectors=[(document_chunk_id, chunk_embedding)])
                
                # Insert document ID and text into MySQL database
                insert_query = "INSERT INTO documents (document_id, text) VALUES (%s, %s)"
                cursor.execute(insert_query, (document_chunk_id, text_chunk))
            
            connection.commit()
    
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
        

if __name__ == '__main__':
    app.run(debug=True)
