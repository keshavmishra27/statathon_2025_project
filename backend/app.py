import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, flash, url_for, redirect, send_file, current_app, session
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.utils import secure_filename
from backend.forms import LoginForm, RegisterForm
from backend import db
from backend.credentials import User
from backend.scores import get_score_for_file
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import KNNImputer
import plotly.express as px
import json
from sklearn.linear_model import LinearRegression
import os
import plotly
from backend.config import Config
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from flask import send_file, current_app
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os
from langchain_ollama import OllamaLLM

# ----------------- Config -----------------
app_blueprint = Blueprint('app_blueprint', __name__)
PROCESSED_FOLDER = "processed"
# ----------------- Helpers ---------------
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_pdf_with_insights(df, workflow_log, weighted_stats, pdf_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    # Setup PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>AI-Enhanced Data Analysis Summary</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # Workflow log
    elements.append(Paragraph("<b>Workflow Log:</b>", styles['Heading2']))
    for log in workflow_log:
        elements.append(Paragraph(f"- {log}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Weighted stats
    if weighted_stats:
        elements.append(Paragraph("<b>Weighted Stats:</b>", styles['Heading2']))
        for k, v in weighted_stats.items():
            elements.append(Paragraph(f"{k}: {v}", styles['Normal']))
        elements.append(Spacer(1, 12))

    # Table snippet (first 10 rows)
    elements.append(Paragraph("<b>Data Preview (first 10 rows):</b>", styles['Heading2']))
    preview = df.head(10).reset_index(drop=True)
    table_data = [preview.columns.tolist()] + preview.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Add numeric distribution plot
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        plt.figure(figsize=(6, 4))
        df[numeric_cols].hist(bins=15, figsize=(8, 6))
        plt.tight_layout()
        chart_path = pdf_path.replace(".pdf", "_chart.png")
        plt.savefig(chart_path)
        plt.close()

        elements.append(Paragraph("<b>Numeric Column Distributions:</b>", styles['Heading2']))
        elements.append(Image(chart_path, width=400, height=300))
        elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)


def ai_schema_suggestion(df):
    suggested_mapping = {}
    suggested_rules = []

    for col in df.columns:
        clean_col = ''.join(filter(str.isalnum, col.lower()))

        # 🔹 Semantic rules
        if "dob" in clean_col or "date" in clean_col:
            suggested_mapping[col] = "Date"
            suggested_rules.append({"column": col, "rule": "valid_date"})
        elif "name" in clean_col:
            suggested_mapping[col] = "FullName"
            suggested_rules.append({"column": col, "rule": "not_empty"})
        elif any(x in clean_col for x in ["sal", "income", "budget"]):
            suggested_mapping[col] = "Income"
            suggested_rules.append({"column": col, "rule": ">= 0"})
        elif "age" in clean_col:
            suggested_mapping[col] = "Age"
            suggested_rules.append({"column": col, "rule": "between", "min": 0, "max": 120})
        else:
            # 🔹 Fallback to data-driven
            if pd.api.types.is_numeric_dtype(df[col]):
                suggested_mapping[col] = "Numeric"
                suggested_rules.append({"column": col, "rule": "numeric"})
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                suggested_mapping[col] = "Date"
                suggested_rules.append({"column": col, "rule": "valid_date"})
            elif df[col].nunique() < 20:
                suggested_mapping[col] = "Categorical"
                suggested_rules.append({"column": col, "rule": "limited_categories"})
            else:
                suggested_mapping[col] = "Text"
                suggested_rules.append({"column": col, "rule": "valid_text"})

    return {
        "schema_mapping": suggested_mapping,
        "rules": suggested_rules,   # ✅ now structured, not just strings
        "imputation_method": "mean",
        "outlier_method": "iqr",
        "weight_column": None,
        "ai_impute": True
    }



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
            upload_folder = current_app.config["UPLOAD_FOLDER"]
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            current_user.upload_count = (current_user.upload_count or 0) + 1
            file_score = get_score_for_file(filename, file_path=file_path, ai_applied=True)
            current_user.score = (current_user.score or 0) + file_score
            db.session.commit()

            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=1)
                else:
                    df = pd.read_excel(file_path, nrows=1)
                suggested_mapping = ai_schema_suggestion(df)
                session['uploaded_columns'] = df.columns.tolist()
                session['ai_schema_suggest'] = suggested_mapping
            except Exception as e:
                flash(f"Could not read columns: {e}", category='warning')
                session['uploaded_columns'] = []
                session['ai_schema_suggest'] = {}

            return redirect(url_for('app_blueprint.configure', filename=filename))

        else:
            flash("Invalid file type. Only CSV/Excel allowed.", category='danger')
            return redirect(request.url)
    return render_template('upload.html')


# ----------------- Configure -----------------
# ------------------ Routes ------------------
@app_blueprint.route("/configure/<filename>", methods=["GET", "POST"])
def configure(filename):
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)


    # Load CSV columns
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()

    # ✅ Example AI suggestions (replace with your real AI call)
    ai_suggest = {
        "schema_mapping": {col: "Numeric" for col in columns},
        "imputation_method": "mean",
        "outlier_method": "zscore",
        "rules": [{"column": col, "rule": "numeric"} for col in columns],
        "weight_column": None,
        "ai_impute": True
    }

    if request.method == "POST":
        action = request.form.get("action")  # which button was pressed

        # collect config selections
        config = {
            "schema_mapping": {col: request.form.get(f"map_{col}", "") for col in columns},
            "imputation_method": request.form.get("imputation_method"),
            "outlier_method": request.form.get("outlier_method"),
            "rules": request.form.getlist("rules"),
            "weight_column": request.form.get("weight_column"),
            "ai_impute": True if request.form.get("ai_impute") else False
        }

        # ✅ Handle visualize button
        if action == "visualize":
            # Do your visualization logic here
            return render_template("visualize.html", filename=filename, config=config, df=df.to_html(classes="table table-bordered"))

        # ✅ Handle analyze button
        elif action == "analyze":
    
            return redirect(url_for("app_blueprint.analyze", filename=filename))

        flash("Unknown action", "danger")
        return redirect(url_for('app_blueprint.configure', filename=filename))


    # GET → render configure page
    return render_template(
        "configure.html",
        filename=filename,
        columns=columns,
        ai_suggest=ai_suggest
    )

llm = OllamaLLM(model="tinyllama")
from langchain_ollama import OllamaLLM

# Initialize Ollama LLM once
llm = OllamaLLM(model="tinyllama")

def generate_ai_insights(df: pd.DataFrame):
    """Generate AI insights using TinyLlama"""
    try:
        prompt = f"""
        You are a data analyst. Summarize the dataset in a clear, human-friendly way.
        Mention trends, anomalies, and structure.

        Dataset preview:
        {df.head(20).to_string()}
        """
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"[TinyLlama error: {e}]"


@app_blueprint.route('/visualize/<filename>')
@login_required
def visualize_page(filename):
    # Prefer session, but fallback to processed folder
    file_path = session.get('data_file')
    if not file_path or not os.path.exists(file_path):
        file_path = os.path.join(PROCESSED_FOLDER, filename)

    if not os.path.exists(file_path):
        flash("Processed file not found. Please re-process your file.", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    df = pd.read_csv(file_path)

    numeric_cols = df.select_dtypes(include='number').columns
    insights = {}

    # Create initial Plotly figure
    fig = px.line(df, y=numeric_cols, title="AI-Augmented Visualization")
    fig.update_layout(template="plotly_dark")

    # Column-wise trend + anomaly detection
    for col in numeric_cols:
        col_data = df[[col]].copy()
        col_data[col] = col_data[col].fillna(col_data[col].mean())

        X = col_data.index.values.reshape(-1, 1)
        y = col_data[col].values

        model = LinearRegression().fit(X, y)
        trend = "increasing" if model.coef_[0] > 0 else "decreasing"

        anomalies = col_data[
            (col_data[col] > col_data[col].mean() + 2 * col_data[col].std()) |
            (col_data[col] < col_data[col].mean() - 2 * col_data[col].std())
        ][col].tolist()

        insights[col] = {
            "trend": trend,
            "anomalies": anomalies
        }

        anomaly_indices = col_data.index[col_data[col].isin(anomalies)]
        if len(anomaly_indices) > 0:
            fig.add_scatter(
                x=anomaly_indices,
                y=col_data.loc[anomaly_indices, col],
                mode="markers",
                marker=dict(color="red", size=10),
                name=f"{col} anomalies"
            )

    # 🔥 AI narrative with safe fallback
    # 🔥 AI narrative using TinyLlama
    ai_narrative = generate_ai_insights(df)

    # If TinyLlama fails for some reason, fall back to local summary
    if not ai_narrative or "TinyLlama error" in ai_narrative:
        ai_narrative = build_ai_summary(df)


    # Serialize Plotly figure
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Send full dataframe to frontend as JSON
    data_json = df.to_json(orient="records")

    return render_template(
        "visualize.html",
        filename=filename,
        graphJSON=graphJSON,
        insights=insights,
        all_columns=df.columns.tolist(),
        ai_narrative=ai_narrative,
        table_html=df.head(10).to_html(classes="table table-striped table-dark table-sm", index=False),
        data_json=data_json
    )

# ----------------- AI-Enhanced Analyze -----------------
@app_blueprint.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    filename = request.args.get('filename')
    if not filename:
        flash("No file to analyze", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        flash("File not found.", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    return run_full_pipeline(file_path, filename)


def build_ai_summary(df: pd.DataFrame):
    ai_summary = {}

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    # Dataset-level stats
    dataset_summary = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "num_numeric": len(numeric_cols),
        "num_categorical": len(categorical_cols),
        "missing_total": int(df.isnull().sum().sum())
    }

    # High-level AI insight
    insight_parts = []
    insight_parts.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    if len(numeric_cols) > 0:
        insight_parts.append(f"{len(numeric_cols)} numeric columns show varying levels of spread.")
    if len(categorical_cols) > 0:
        insight_parts.append(f"{len(categorical_cols)} categorical columns capture distinct labels.")
    if dataset_summary["missing_total"] > 0:
        insight_parts.append(f"There are {dataset_summary['missing_total']} missing values across the dataset.")

    dataset_summary["insight"] = " ".join(insight_parts)

    # Column-wise AI summary
    for col in df.columns:
        col_data = df[col]
        missing_count = int(col_data.isnull().sum())

        if pd.api.types.is_numeric_dtype(col_data):
            mean_val = float(col_data.mean()) if not col_data.isnull().all() else None
            std_val = float(col_data.std()) if not col_data.isnull().all() else None

            if mean_val is not None and std_val is not None:
                if std_val < 0.1 * mean_val:
                    variability = "low variability"
                elif std_val < 0.5 * mean_val:
                    variability = "moderate variability"
                else:
                    variability = "high variability"
                insight = f"Numeric column with {variability}, mean ≈ {mean_val:.2f}, std ≈ {std_val:.2f}. Missing: {missing_count}."
            else:
                insight = f"Numeric column but mostly missing values ({missing_count} missing)."

            ai_summary[col] = {
                "type": "numeric",
                "mean": mean_val,
                "std": std_val,
                "missing": missing_count,
                "insight": insight
            }

        else:
            unique_count = int(col_data.nunique(dropna=True))
            mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else None

            if mode_val:
                insight = f"Categorical column with {unique_count} unique values. Most common value is '{mode_val}'. Missing: {missing_count}."
            else:
                insight = f"Categorical column with {unique_count} unique values but no clear mode. Missing: {missing_count}."

            ai_summary[col] = {
                "type": "categorical",
                "unique": unique_count,
                "mode": str(mode_val) if mode_val is not None else None,
                "missing": missing_count,
                "insight": insight
            }

    return dataset_summary, ai_summary


def run_full_pipeline(file_path, filename):
    df = pd.read_csv(file_path)

    workflow_log = [
        f"File '{filename}' successfully loaded.",
        f"DataFrame shape: {df.shape}",
        f"Columns detected: {list(df.columns)}"
    ]

    weighted_stats = {col: float(df[col].mean()) for col in df.select_dtypes(include='number').columns}

    # Dataset + Column-level AI summary
    dataset_summary, ai_summary = build_ai_summary(df)

    # 🔥 Add Gemini AI summary
    ai_summary_text = generate_ai_insights(df)

    table_html = df.head(10).to_html(classes="table table-dark table-striped", index=False)
    processed_filename = f"processed_{secure_filename(filename)}"

    os.makedirs("uploads", exist_ok=True)
    processed_path = os.path.join("uploads", processed_filename)
    df.to_csv(processed_path, index=False)

    return render_template(
        "result.html",
        filename=filename,
        workflow_log=workflow_log,
        weighted_stats=weighted_stats,
        dataset_summary=dataset_summary,
        ai_summary=ai_summary,
        ai_summary_text=ai_summary_text,   # ✅ new
        table_html=table_html,
        processed_filename=processed_filename
    )


def make_columns_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" for i in range(sum(cols == dup))]
    return cols.tolist()


@app_blueprint.route('/leaderboard')
@login_required
def leaderboard_page():
    users = User.query.order_by(User.score.desc(), User.upload_count.desc()).all()
    current_rank = None
    for idx, user in enumerate(users, start=1):
        if user.id == current_user.id:
            current_rank = idx
            break
    return render_template('leaderboard.html', users=users, current_rank=current_rank)


# 🔹 Upload file
@app_blueprint.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        session["uploaded_file"] = filename
        return redirect(url_for("app_blueprint.analyze", filename=filename))

    return render_template("upload.html")


@app_blueprint.route("/download_pdf/<filename>")
def download_pdf(filename):
    upload_folder = current_app.config['UPLOAD_FOLDER']
    processed_folder = current_app.config['PROCESSED_FOLDER']
    plots_dir = os.path.join(processed_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        flash("File not found.", "danger")
        return redirect(url_for("app_blueprint.upload_page"))

    # 🔹 Load dataset
    df = pd.read_csv(file_path)
    dataset_summary, ai_summary = build_ai_summary(df)

    # 🔹 AI narrative (TinyLlama / fallback)
    try:
        ai_summary_text = generate_ai_insights(df)
        if not ai_summary_text or "error" in ai_summary_text.lower():
            raise ValueError("AI summary invalid")
    except Exception as e:
        print(f"[WARN] AI model failed: {e}")
        ai_summary_text = (
            "This dataset contains "
            f"{dataset_summary['num_rows']} rows and {dataset_summary['num_cols']} columns. "
            f"There are {dataset_summary['num_numeric']} numeric columns, "
            f"{dataset_summary['num_categorical']} categorical columns, "
            f"and {dataset_summary['missing_total']} missing values. "
            "Overall, the dataset is structured and ready for statistical analysis."
        )


    # 🔹 Workflow + stats
    workflow_log = [
        f"File '{filename}' successfully loaded.",
        f"DataFrame shape: {df.shape}",
        f"Columns detected: {list(df.columns)}"
    ]
    weighted_stats = {col: float(df[col].mean()) for col in df.select_dtypes(include='number').columns}

    # 🔹 PDF path
    pdf_path = os.path.join(processed_folder, f"{filename}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

# ------------------- PAGE 1: EXECUTIVE SUMMARY -------------------
    story.append(Paragraph("📄 Executive AI Summary", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Dataset:</b> {filename}", styles["Heading3"]))
    story.append(Spacer(1, 12))

    summary_points = [
    f"Shape: {dataset_summary['num_rows']} rows × {dataset_summary['num_cols']} columns",
    f"Numeric Columns: {dataset_summary['num_numeric']}",
    f"Categorical Columns: {dataset_summary['num_categorical']}",
    f"Missing Values: {dataset_summary['missing_total']}",
]
    for point in summary_points:
        story.append(Paragraph("• " + point, styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("🤖 AI Narrative", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(ai_summary_text, styles["Normal"]))
    story.append(PageBreak())

    # ------------------- DATASET OVERVIEW -------------------
    story.append(Paragraph("📂 Dataset Overview", styles["Heading2"]))
    data = [
        ["Rows", dataset_summary["num_rows"]],
        ["Columns", dataset_summary["num_cols"]],
        ["Numeric Columns", dataset_summary["num_numeric"]],
        ["Categorical Columns", dataset_summary["num_categorical"]],
        ["Missing Values", dataset_summary["missing_total"]],
    ]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table)
    story.append(PageBreak())

    # ------------------- WORKFLOW LOG -------------------
    story.append(Paragraph("📝 Workflow Log", styles["Heading2"]))
    for step in workflow_log:
        story.append(Paragraph(f"• {step}", styles["Normal"]))
    story.append(PageBreak())

    # ------------------- WEIGHTED STATS -------------------
    story.append(Paragraph("📈 Weighted Statistics", styles["Heading2"]))
    stats_data = [[k, f"{v:.2f}"] for k, v in weighted_stats.items()]
    stats_table = Table([["Metric", "Value"]] + stats_data)
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(stats_table)
    story.append(PageBreak())

    # ------------------- VISUALIZATIONS -------------------
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        story.append(Paragraph("📊 Visual Diagnostics", styles["Heading2"]))

        # Histogram
        plt.figure(figsize=(8, 6))
        df[numeric_cols].hist(bins=20, figsize=(10, 8))
        plt.tight_layout()
        hist_path = os.path.join(plots_dir, f"{filename}_hist.png")
        plt.savefig(hist_path)
        plt.close()
        story.append(Paragraph("Distribution of Numeric Columns", styles["Heading3"]))
        story.append(Image(hist_path, width=400, height=300))
        story.append(Spacer(1, 15))

        # Boxplot
        plt.figure(figsize=(10, 6))
        df[numeric_cols].plot(kind="box")
        plt.title("Boxplot for Outlier Detection")
        box_path = os.path.join(plots_dir, f"{filename}_box.png")
        plt.savefig(box_path)
        plt.close()
        story.append(Paragraph("Boxplot for Outlier Detection", styles["Heading3"]))
        story.append(Image(box_path, width=400, height=300))
        story.append(Spacer(1, 15))

        # Correlation Heatmap
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(plots_dir, f"{filename}_corr.png")
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        story.append(Paragraph("Correlation Heatmap", styles["Heading3"]))
        story.append(Image(heatmap_path, width=400, height=300))
        story.append(Spacer(1, 15))
        story.append(PageBreak())

    # ------------------- BUILD PDF -------------------
    doc.build(story)
    return send_file(pdf_path, as_attachment=True)







