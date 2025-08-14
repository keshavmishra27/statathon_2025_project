# backend/scores.py

# Score per file type
FILE_TYPE_SCORES = {
    "csv": 10,
    "xlsx": 5,
    "xls": 5
}

def get_score_for_file(filename):
    """
    Returns score based on file extension.
    """
    ext = filename.rsplit('.', 1)[1].lower()
    return FILE_TYPE_SCORES.get(ext, 0)
