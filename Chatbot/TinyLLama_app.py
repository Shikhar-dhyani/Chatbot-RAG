
from flask import Flask, request, jsonify, redirect, url_for, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import mysql.connector
from mysql.connector import Error
from werkzeug.utils import secure_filename
import shutil
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from PyPDF2 import PdfReader


UPLOAD_FOLDER = 'uploads'
SAVED_FILES='archived_pdfs'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

app = Flask(__name__)

# Flask configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SAVED_FILES']='archived_pdfs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# User Loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def save_embeddings(texts,base_document_id):
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
                document_id = f"{base_document_id}_{i}"
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

def get_answer(input_text):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Update this to your actual model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=200)

    # Generate a response
    output_ids = model.generate(
    input_ids,
    max_length=270,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.1,  # Adjust for diversity vs confidence
    top_k=20,  # Top-k sampling
    top_p=0.95,  # Nucleus sampling
    repetition_penalty=1.2,  # Adjust repetition penalty
    )

# Decode the output to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response


@app.route('/')
@login_required
def index():
    return "Welcome to the Dashboard!"


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter(User.username == username).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')


@app.route('/admin/create_user', methods=['GET', 'POST'])
@login_required
def create_user():
    if not current_user.role == 'admin':
        return "Unauthorized", 403  # Redirect or return a different response for unauthorized access
    if request.method == 'POST':
        # Using request.form to access form data
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')

        if not all([username, password, role]):
            flash('Missing data', 'error')
            return redirect(url_for('create_user'))
        if db.session.query(User.id).filter_by(username=username).first():
            flash('User already exists', 'error')
            return redirect(url_for('create_user'))
        
        user = User(username=username, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('User created successfully', 'success')
        return redirect(url_for('index'))  # Redirect to a confirmation page or back to the form
    # Display the form for a GET request
    return render_template('create_user.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file_or_uri():
    # Check for file upload first
    if not current_user.role == 'admin':
        return "Unauthorized", 403  # Redirect or return a different response for unauthorized access
    files = request.files.getlist('file')
    uri = request.form.get('uri', '')  # URI input from form data
    
    if not files and not uri:
        return jsonify(error="No file or URI provided"), 400
    
    
    uploaded_files, skipped_files,errors = [], [],[]
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            skipped_files.append(file.filename)
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        main_path= os.path.join(app.config['SAVED_FILES'], filename)

        if os.path.exists(main_path):
            skipped_files.append(filename)
            continue

        if os.path.exists(save_path):
            skipped_files.append(filename)
            continue

        file.save(save_path)
        uploaded_files.append(filename)

    if not files and uri:
        try:
            filename = uri.split('/')[-1]
            if filename == '' or not allowed_file(filename):
                errors.append("Invalid file type or empty filename from URI")
            else:
                filename = secure_filename(filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                if os.path.exists(save_path):
                    skipped_files.append(filename)
                else:
                    response = requests.get(uri)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        uploaded_files.append(filename)
                    else:
                        errors.append(f"Failed to download the file from URI. Status code: {response.status_code}")
        except Exception as e:
            errors.append(f"Error downloading file from URI: {str(e)}")

    return jsonify(uploaded=uploaded_files, skipped=skipped_files), 200




@app.route('/load')
@login_required
def load():
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized access'}), 403
    
    file_directory = "uploads/"
    archive_directory = "archived_files/"

    if not os.path.exists(archive_directory):
        os.makedirs(archive_directory)

    files = os.listdir(file_directory)
    allowed_files = [file for file in files if file.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS]

    for file_name in allowed_files:
        file_path = os.path.join(file_directory, file_name)
        file_extension = file_name.rsplit('.', 1)[-1].lower()
        base_document_id = os.path.splitext(file_name)[0]
        
        if file_extension == 'pdf':
            reader = PdfReader(file_path)
            pages = [page.extract_text() for page in reader.pages]
            context = "\n".join(pages)
        elif file_extension in {'doc', 'docx'}:
            import docx
            doc = docx.Document(file_path)
            context = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                context = file.read()

        # Assuming you have a common text processing function for all types
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
        texts = text_splitter.split_text(context)
        save_embeddings(texts, base_document_id)

        shutil.move(file_path, os.path.join(archive_directory, file_name))

    return jsonify({'Response': "Loaded"})




@app.route('/response_query', methods=['POST'])
@login_required
def response_query():
    document_name = request.form.get('document_name')
    query_text = request.form.get('query_text')

    if not document_name or not query_text:
        return "Missing data", 400 
    pinecone_client = Pinecone(api_key="e15f4903-51b8-4ed9-9489-cf7883edb8ab")
    pinecone_index = pinecone_client.Index("demo")
    query_embedding = generate_embeddings([query_text])[0].tolist()
    query_results = pinecone_index.query(vector=query_embedding, top_k=20)  # Increase top_k if needed
    # Filter the results to only include those matching the document_name
    ids_to_lookup = [match['id'] for match in query_results["matches"] if match['id'].startswith(document_name + "_")]

    if not ids_to_lookup:
        return jsonify({"Response": "No matches found for this document"}), 404

    try:
        connection = mysql.connector.connect(
        host='localhost',  
        user='root', 
        password='Clicflyer@123', 
        database='Policy'  
        )
        if connection.is_connected():
            id_to_text = fetch_texts_for_ids(connection, ids_to_lookup)
            context = "\n".join(id_to_text.get(id, " ") for id in ids_to_lookup)
            input_text = f"question: {query_text} context: {context}"
            answer = get_answer(input_text)  # Assuming this function generates a summary for each chunk
            return jsonify({"Response": answer})
    finally:
        if connection and connection.is_connected():
            connection.close()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)