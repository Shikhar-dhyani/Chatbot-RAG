import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone
import mysql.connector
from mysql.connector import Error

UPLOAD_FOLDER = 'uploads'  # Change this to your desired upload directory
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # It's important for Flask sessions and CSRF protection
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def Generate_embeddings(file):
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
            
            filename = secure_filename(file.filename)
            file_to_be_saved = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_to_be_saved)

            pdf_loader = PyPDFLoader(file_to_be_saved)
            pages = pdf_loader.load_and_split()

            # Generate a document ID and embeddings for each page
            base_document_id = os.path.splitext(filename)[0]  # Remove '.pdf' from filename
            for i, page in enumerate(pages, start=1):
                
                page_text = page.page_content

                embedding = generate_embeddings([page_text])[0].tolist()  # Generate embedding for the page
                document_id = f"{base_document_id}_{i}"
                pinecone_index.upsert(vectors=[(document_id, embedding)])
                
                
                insert_query = "INSERT INTO documents (document_id, text) VALUES (%s, %s)"
                cursor.execute(insert_query, (document_id, page_text))
            
            connection.commit()

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file and allowed_file(file.filename):
        Generate_embeddings(file)
        return jsonify(message=f"File {file.filename} uploaded successfully"), 200
    else:
        return jsonify(error="File type not allowed"), 400

if __name__ == '__main__':
    app.run(debug=True)
