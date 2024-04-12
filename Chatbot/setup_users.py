from authorization import app, db
from authorization import User


def create_admin_user(username, password):
    with app.app_context():
        # Check if the user already exists
        if User.query.filter_by(username=username).first():
            print('Admin user already exists.')
            return
        
        # Create a new admin user
        admin = User(username=username, role='admin')
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        print('Admin user created successfully.')

if __name__ == '__main__':
    # Call the function with your desired admin username and password
    create_admin_user('admin', 'Clicflyer')
