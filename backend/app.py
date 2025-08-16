import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, flash, url_for, redirect, send_file, current_app, session
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from backend.forms import LoginForm, RegisterForm
from backend import db
from backend.credentials import User
from backend.scores import get_score_for_file
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

# ----------------- Config -----------------
app_blueprint = Blueprint('app_blueprint', __name__)
UPLOAD_FOLDER = os.path.join("backend", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# ----------------- Helpers -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_pdf_from_df(df, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    text = c.beginText(40, height - 40)
    text.setFont("Helvetica", 10)
    for line in df.to_string(index=False).split('\n'):
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()

# ----------------- Routes -----------------
@app_blueprint.route('/')
@app_blueprint.route('/home')
def home_page():
    if current_user.is_authenticated:
        flash(f"Welcome back, {current_user.username}!", category='info')
    return render_template('home.html')

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

@app_blueprint.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    flash('Please log in', category='info')
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
@login_required
def logout_page():
    logout_user()
    flash('Logged out successfully!', category='info')
    return redirect(url_for('app_blueprint.home_page'))

# ----------------- Upload -----------------
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

            current_user.upload_count = (current_user.upload_count or 0) + 1
            file_score = get_score_for_file(filename, file_path=file_path, ai_applied=False)
            current_user.score = (current_user.score or 0) + file_score
            db.session.commit()

            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=1)
                else:
                    df = pd.read_excel(file_path, nrows=1)
                session['uploaded_columns'] = df.columns.tolist()
            except Exception as e:
                flash(f"Could not read columns for auto mapping: {e}", category='warning')
                session['uploaded_columns'] = []

            return redirect(url_for('app_blueprint.configure_processing', filename=filename))
        else:
            flash("Invalid file type. Only CSV/Excel allowed.", category='danger')
            return redirect(request.url)
    return render_template('upload.html')

# ----------------- Configure -----------------
@app_blueprint.route('/configure/<filename>', methods=['GET', 'POST'])
@login_required
def configure_processing(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    columns = session.get('uploaded_columns', df.columns.tolist())
    if request.method == 'POST':
        config = {
            "schema_mapping": {col: request.form.get(f"map_{col}") for col in columns},
            "imputation_method": request.form.get("imputation_method"),
            "outlier_method": request.form.get("outlier_method"),
            "rules": request.form.getlist("rules"),
            "weight_column": request.form.get("weight_column"),
            "ai_impute": True if request.form.get("ai_impute") == "on" else False
        }
        session['processing_config'] = config
        processed_file_path = os.path.join(PROCESSED_FOLDER, f"{filename}_processed.json")
        df.to_json(processed_file_path, orient="records")
        session['data_file'] = processed_file_path
        if request.form.get("action") == "visualize":
            return redirect(url_for('app_blueprint.visualize_page'))
        else:
            return redirect(url_for('app_blueprint.analyze', filename=filename))
    return render_template('configure.html', filename=filename, columns=columns)

# ----------------- AI-Enhanced Analyze -----------------
@app_blueprint.route('/analyze', methods=['GET', 'POST'])
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
    return run_full_pipeline(file_path, filename)

def run_full_pipeline(file_path, filename):
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from scipy import stats
    from scipy.stats.mstats import winsorize
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    config = session.get('processing_config', {})
    workflow_log = []

    # Schema Mapping
    schema_map = config.get("schema_mapping", {})
    for old_col, new_col in schema_map.items():
        if new_col and old_col != new_col:
            df.rename(columns={old_col: new_col}, inplace=True)
            workflow_log.append(f'Mapped "{old_col}" to "{new_col}"')

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Missing Value Imputation
    if config.get("ai_impute") and numeric_cols:
        workflow_log.append("AI Imputation Enabled: Using RandomForestRegressor")
        for col in numeric_cols:
            missing_idx = df[df[col].isnull()].index
            if not missing_idx.empty:
                X_train = df[numeric_cols].drop(col, axis=1)
                y_train = df[col]
                known_mask = y_train.notnull()
                if known_mask.sum() > 1:
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_train[known_mask], y_train[known_mask])
                    df.loc[missing_idx, col] = rf.predict(X_train.loc[missing_idx])
        workflow_log.append("AI Imputation completed")
    else:
        impute_method = config.get("imputation_method", "mean")
        if impute_method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            workflow_log.append("Missing values filled with Mean")
        elif impute_method == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            workflow_log.append("Missing values filled with Median")
        elif impute_method == "knn":
            imputer = KNNImputer(n_neighbors=3)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            workflow_log.append("Missing values filled with KNN Imputer (k=3)")

    # Outlier Detection
    outlier_method = config.get("outlier_method", "isolationforest")
    if numeric_cols:
        if outlier_method == "isolationforest":
            iso = IsolationForest(contamination=0.05, random_state=42)
            df['Outlier'] = iso.fit_predict(df[numeric_cols])
            df['Outlier'] = df['Outlier'].map({1:'Normal', -1:'Outlier'})
            workflow_log.append("Outliers detected using Isolation Forest")
        elif outlier_method == "iqr":
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df[numeric_cols] < (Q1 - 1.5*IQR)) | (df[numeric_cols] > (Q3 + 1.5*IQR))).any(axis=1)
            df['Outlier'] = np.where(mask, 'Normal','Outlier')
            workflow_log.append("Outliers detected using IQR method")
        elif outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(df[numeric_cols]))
            mask = (z_scores<3).all(axis=1)
            df['Outlier'] = np.where(mask,'Normal','Outlier')
            workflow_log.append("Outliers detected using Z-Score method")
        elif outlier_method == "winsorization":
            for col in numeric_cols:
                df[col] = winsorize(df[col], limits=[0.05,0.05])
            df['Outlier'] = 'Adjusted via Winsorization'
            workflow_log.append("Outliers handled using Winsorization (5% limits)")

    # Rule Validation
    rules = config.get("rules", [])
    for rule in rules:
        if rule=="age_limit" and "Age" in df.columns:
            invalid_count = (df["Age"]>120).sum()
            workflow_log.append(f"Rule 'Age < 120': {invalid_count} violations")
        if rule=="income_positive" and "Income" in df.columns:
            invalid_count = (df["Income"]<=0).sum()
            workflow_log.append(f"Rule 'Income > 0': {invalid_count} violations")

    # Weighted Stats
    weighted_stats = {}
    weight_col = config.get("weight_column")
    if weight_col and weight_col in df.columns and 'Price' in df.columns:
        w = df[weight_col]
        p = df['Price']
        weighted_mean = np.average(p, weights=w)
        std_error = np.sqrt(np.cov(p, aweights=w)/len(p))
        margin_error = 1.96*std_error
        weighted_stats = {'Weighted Mean Price': round(weighted_mean,2), 'Margin of Error': round(margin_error,2)}
        workflow_log.append(f"Applied weights from '{weight_col}' and calculated MOE")

    # AI Summary
    ai_summary = {col: {"mean": round(df[col].mean(),2), "std": round(df[col].std(),2), "missing": int(df[col].isnull().sum())} for col in numeric_cols} if numeric_cols else {}

    session['data'] = df.to_dict(orient="list")
    table_html = df.head(20).to_html(classes="table table-bordered table-striped", index=False)
    flash("Data processed successfully with AI-enhanced pipeline!", category='success')
    return render_template('result.html', filename=filename, table_html=table_html, weighted_stats=weighted_stats, workflow_log=workflow_log, ai_summary=ai_summary)

# ----------------- Visualize -----------------
@app_blueprint.route('/visualize')
@login_required
def visualize_page():
    if 'data_file' not in session:
        flash("No data available. Upload and configure first.", category="danger")
        return redirect(url_for('app_blueprint.upload_page'))
    df = pd.read_json(session['data_file'])
    df_json = df.to_dict(orient='list')
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    ai_summary = {col: {"mean": round(df[col].mean(),2), "std": round(df[col].std(),2), "missing": int(df[col].isnull().sum())} for col in numeric_cols}
    return render_template('visualize.html', df_json=df_json, columns=df.columns.tolist(), ai_summary=ai_summary)

# ----------------- Download -----------------
@app_blueprint.route('/download/<filename>')
@login_required
def download_pdf(filename):
    pdf_path = os.path.join(current_app.root_path, "static", "uploads", filename)
    return send_file(pdf_path, as_attachment=True)

# ----------------- Leaderboard -----------------
@app_blueprint.route('/leaderboard')
@login_required
def leaderboard_page():
    users = User.query.order_by(User.score.desc()).all()
    current_rank = next((i+1 for i,u in enumerate(users) if u.id==current_user.id), None)
    return render_template('leaderboard.html', users=users, current_rank=current_rank)
