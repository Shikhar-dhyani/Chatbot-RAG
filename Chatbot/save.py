from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
# Additional imports as needed
from celery import Celery

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Flask configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['SECRET_KEY'] = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# celery = Celery(app.name, broker='redis://localhost:6379/0')
# celery.conf.update(
#     result_backend='redis://localhost:6379/0',
    
# )


# @celery.task
# def process_pdf_and_generate_embeddings(filename, save_path):
#     pdf_loader = PyPDFLoader(save_path)
#     pages = pdf_loader.load_and_split()
#     document_text = "\n".join([page.page_content for page in pages])
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=400)
#     texts = text_splitter.split_text(document_text)
    
#     base_document_id = os.path.splitext(filename)[0]
    
#     for i, text_chunk in enumerate(texts, start=1):
#         chunk_embedding = generate_embeddings([text_chunk])[0].tolist()
#         document_chunk_id = f"{base_document_id}_chunk_{i}"
#         # print(document_chunk_id)
        


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     files = request.files.getlist('file')
    
#     if not files:
#         return jsonify(error="No files part"), 400
    
#     filenames = []
#     skipped_files = []
#     for file in files:
#         if file.filename == '':
#             continue  # Skip empty files
#         if not allowed_file(file.filename):
#             return jsonify(error=f"File {file.filename} type not allowed"), 400

#         filename = secure_filename(file.filename)
#         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#         # Check for duplicate file name
#         if os.path.exists(save_path):
#             skipped_files.append(filename)
#             continue  # Skip saving this file
        
#         file.save(save_path)
#         filenames.append(filename)
    
#     if not filenames and not skipped_files:
#         return jsonify(error="No valid files selected"), 400
#     message = "Files uploaded successfully" if filenames else "No new files uploaded"
#     if skipped_files:
#         message += f". Skipped duplicates: {', '.join(skipped_files)}"
#     return jsonify(message=message, uploaded_files=filenames, skipped_files=skipped_files), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    
    if not files:
        return jsonify(error="No files part"), 400
    
    uploaded_files, skipped_files = [], []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            skipped_files.append(file.filename)
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(save_path):
            skipped_files.append(filename)
            continue

        file.save(save_path)
        uploaded_files.append(filename)
        # process_pdf_and_generate_embeddings.delay(filename, save_path)

    return jsonify(uploaded=uploaded_files, skipped=skipped_files), 200


if __name__ == '__main__':
    app.run(debug=True)
