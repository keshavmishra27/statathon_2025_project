from backend import create_app, db
from backend.credentials import User

app = create_app()

with app.app_context():
    db.create_all()
    print(" Tables created!")