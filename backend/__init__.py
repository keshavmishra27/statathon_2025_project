import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
bcrypt = Bcrypt()

def create_app():
    # Load .env from project root (same as run.py)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "super_secret_123")
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///yourdatabase.db")

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    CORS(app)

    # Flask-Login config
    login_manager.login_view = "app_blueprint.login_page"

    # ✅ Configure Gemini API (from .env)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GOOGLE_API_KEY not found. Did you create a .env file?")
    genai.configure(api_key=api_key)

    # Register blueprints
    from backend.app import app_blueprint
    app.register_blueprint(app_blueprint)

    return app
