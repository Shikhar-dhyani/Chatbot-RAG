from flask import Flask, request, jsonify, redirect, url_for, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'random-key-for-security-can-change-later'
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

@app.route('/load')
@login_required
def load():
    if not current_user.role == 'admin':
        return jsonify({'error': 'Unauthorized access'}), 403
    return jsonify({'message': 'Dummy load function. Access granted to admin.'})

@app.route('/response')
def response():
    return jsonify({'message': 'Dummy response function. Access granted to anyone.'})

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


@app.route('/')
@login_required
def index():
    return "Welcome to the Dashboard!"


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
