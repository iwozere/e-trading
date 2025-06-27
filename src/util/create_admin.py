import getpass
import os

from flask import Flask
from src.management.webgui.models import User, db

# Set up Flask app and DB config (must match your webgui app)
app = Flask(__name__)
db_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "db", "webgui_users.db"
)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    username = input("Enter admin username: ")
    email = input("Enter admin email: ")
    password = getpass.getpass("Enter admin password: ")
    user = User.query.filter_by(username=username).first()
    if user:
        print(f"User '{username}' already exists.")
        update = (
            input(
                "Do you want to update this user to admin and/or reset password? (y/N): "
            )
            .strip()
            .lower()
        )
        if update == "y":
            user.is_admin = True
            reset_pw = (
                input("Do you want to reset the password? (y/N): ").strip().lower()
            )
            if reset_pw == "y":
                user.set_password(password)
                print("Password updated.")
            db.session.commit()
            print(f"User '{username}' is now admin.")
        else:
            print("No changes made.")
    else:
        user = User(username=username, email=email, is_admin=True)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        print(f"Admin user '{username}' created successfully!")
