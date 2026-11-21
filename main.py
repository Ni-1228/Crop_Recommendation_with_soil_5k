from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

NUM_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CAT_COLS = ["soil"]
FEATURE_COLS = NUM_COLS + CAT_COLS


def build_input_df(payload: dict) -> pd.DataFrame:
  row = {}
  for col in NUM_COLS:
    row[col] = float(payload[col])
  for col in CAT_COLS:
    row[col] = str(payload[col]).strip()
  return pd.DataFrame([row])[FEATURE_COLS]


@app.route("/", methods=["GET", "POST"])
def home():
  crop = None
  if request.method == "POST":
    df_input = build_input_df(request.form)
    crop = model.predict(df_input)[0]
  return render_template("index.html", crop=crop)


@app.route("/predict", methods=["POST"])
def predict():
  data = request.get_json()
  df_input = build_input_df(data)
  prediction = model.predict(df_input)[0]
  return jsonify({"crop": prediction})


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)