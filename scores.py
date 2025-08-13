# central place for scoring logic
SCORES = {
  "Biodegradable": 10,
  "Ewaste": 20,
  "hazardous": 30,
  "Non Biodegradable": 40,
  "Pharmaceutical and Biomedical Waste": 50
}

def get_score_for_class(predicted_class, count):
    return SCORES.get(predicted_class, 0) * count
