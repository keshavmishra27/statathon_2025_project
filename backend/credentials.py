import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, flash, url_for, redirect, send_file
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from backend.forms import LoginForm, RegisterForm
from backend import db
from backend.credentials import User

app_blueprint = Blueprint('app_blueprint', __name__)
UPLOAD_FOLDER = os.path.join("backend", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_pdf_from_df(df, pdf_path):
    """Convert DataFrame to PDF file."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    text = c.beginText(40, height - 40)
    text.setFont("Helvetica", 10)

    for line in df.to_string(index=False).split('\n'):
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()

# ---------------- HOME ----------------
@app_blueprint.route('/')
@app_blueprint.route('/home')
def home_page():
    if current_user.is_authenticated:
        flash(f"Welcome back, {current_user.username}!", category='info')
    return render_template('home.html')

# ---------------- UPLOAD ----------------
@app_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in request.", category='danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", category='danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Increment upload_count
            current_user.upload_count = (current_user.upload_count or 0) + 1
            db.session.commit()

            return redirect(url_for('app_blueprint.analyze', filename=filename))
        else:
            flash("Invalid file type. Only CSV/Excel allowed.", category='danger')
            return redirect(request.url)

    return render_template('upload.html')

# ---------------- ANALYZE ----------------
@app_blueprint.route('/analyze')
@login_required
def analyze():
    filename = request.args.get('filename')
    if not filename:
        flash("No file to analyze", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        flash("File not found.", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    # Load file
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Data preprocessing
    df = df.dropna(how='all')
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna("Unknown")

    # Save processed PDF
    pdf_path = os.path.join(UPLOAD_FOLDER, f'{os.path.splitext(filename)[0]}_processed.pdf')
    create_pdf_from_df(df, pdf_path)

    flash("Data processed successfully! Download PDF below.", category='success')
    return send_file(pdf_path, as_attachment=True)

# ---------------- LEADERBOARD ----------------
@app_blueprint.route('/leaderboard')
@login_required
def leaderboard_page():
    users = User.query.order_by(User.upload_count.desc()).all()
    current_rank = next((i + 1 for i, u in enumerate(users) if u.id == current_user.id), None)
    return render_template('leaderboard.html', users=users, current_rank=current_rank)

# ---------------- REGISTER ----------------
@app_blueprint.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(
            username=form.username.data,
            email_address=form.email_address.data,
            password=form.pswd.data,
            score=0,
            upload_count=0
        )
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f"Account created successfully. You are now logged in as {user_to_create.username}", category='success')
        return redirect(url_for('app_blueprint.home_page'))

    if form.errors:
        for err_msg in form.errors.values():
            flash(err_msg, category='danger')
    return render_template('register.html', form=form)

# ---------------- LOGIN ----------------
@app_blueprint.route('/login', methods=['GET', 'POST'])
def login_page():
    flash('Please log in', category='info')
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_pswd_correction(form.pswd.data):
            login_user(attempted_user)
            flash('Logged in successfully!', category='info')
            return redirect(url_for('app_blueprint.home_page'))
        else:
            flash('Invalid username or password!', category='danger')
    return render_template('login.html', form=form)

# ---------------- LOGOUT ----------------
@app_blueprint.route('/logout')
@login_required
def logout_page():
    logout_user()
    flash('Logged out successfully!', category='info')
    return redirect(url_for('app_blueprint.home_page'))

