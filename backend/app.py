from flask import Blueprint, render_template, request, flash, url_for, redirect
from flask_login import login_required, current_user, login_user,logout_user
from werkzeug.utils import secure_filename
import os
from backend.class_pred import classify_image
from backend.forms import LoginForm, RegisterForm
from backend.obj_count import detect_objects_and_save
from backend.scores import SCORES
from backend import db
from backend.credentials import User

app_blueprint = Blueprint('app_blueprint', __name__)
UPLOAD_FOLDER = os.path.join("backend", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app_blueprint.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'images' not in request.files:
        flash("No file part", 'danger')
        return redirect(url_for('app_blueprint.upload_page'))

    files = request.files.getlist('images')
    results = []

    for file in files:
        if not file or file.filename == '' or not allowed_file(file.filename):
            continue

        fn = secure_filename(file.filename)
        in_path  = os.path.join(UPLOAD_FOLDER, "in_"  + fn)
        out_path = os.path.join(UPLOAD_FOLDER, "out_" + fn)
        file.save(in_path)

        # 1) Classify + annotate label
        predicted_class, confidence = classify_image(in_path, out_path)

        # 2) Detect objects + draw boxes
        object_count = detect_objects_and_save(out_path, out_path)

        # 3) Compute per-image score
        per_item_score = SCORES.get(predicted_class, 0)
        image_score    = per_item_score * object_count

        # 4) Award to user
        current_user.score += image_score

        results.append({
            'filename': fn,
            'predicted_class': predicted_class,
            'confidence': round(confidence*100,1),
            'object_count': object_count,
            'image_score': image_score,
            'url': url_for('static', filename='uploads/' + "out_" + fn)
        })

    db.session.commit()
    return render_template('result.html', results=results)

@app_blueprint.route('/')
@app_blueprint.route('/home')
def home_page():
    if current_user.is_authenticated:
        flash(f"Welcome back, {current_user.username}!", category='info')
    return render_template('home.html')

@app_blueprint.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')

@app_blueprint.route('/leaderboard')
def leaderboard_page():
    users = User.query.order_by(User.score.desc()).all()
    return render_template('leaderboard.html', users=users)


@app_blueprint.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(
            username=form.username.data,
            email_address=form.email_address.data,
            password=form.pswd.data,
            score=0  # Initial score set to 0
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


@app_blueprint.route('/logout')
def logout_page():
    logout_user()
    flash('Logged out successfully!', category='info')
    return redirect(url_for('app_blueprint.home_page'))
