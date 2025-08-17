import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-key")  
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit

    # Add other future configs (DB, AI models, etc.)
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///site.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
