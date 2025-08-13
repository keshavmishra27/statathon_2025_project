from backend import db, login_manager, bcrypt
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model,UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String(length=27),unique=True,nullable=False)
    email_address=db.Column(db.String(length=50),unique=True,nullable=False)
    pswd_hash=db.Column(db.String(length=60),nullable=False)
    score=db.Column(db.Integer, nullable=False, default=0)

    @property
    def password(self):
        raise AttributeError("Password is write-only.")
    
    @password.setter
    def password(self, plain_test_pswd):
        self.pswd_hash = bcrypt.generate_password_hash(plain_test_pswd).decode('utf-8')

    def check_pswd_correction(self, attempted_pswd):
        return bcrypt.check_password_hash(self.pswd_hash, attempted_pswd)