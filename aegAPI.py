from flask import Flask, jsonify, request
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            essay = data["essay"]

            pipeline = joblib.load("model.pkl")
        except ValueError:
            return jsonify("Please enter an essay.")

        return jsonify(pipeline.predict(essay).tolist())

@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            training_set = joblib.load("training_data.pkl")
            training_labels = joblib.load("training_labels.pkl")

            df = pd.read_json(data)

            df_training_set = df['essay']
            df_training_labels = df["score"]

            df_training_set = pd.concat([training_set, df_training_set])
            df_training_labels = pd.concat([training_labels, df_training_labels])

            pipeline = Pipeline([('tfidf', TfidfVectorizer(min_df = 0)),  # integer counts to weighted TF-IDF scores
            ('classifier', XGBClassifier(n_estimators= 150, random_state= 1)),  # train on TF-IDF vectors w/ XGB classifier
            ])

            pipeline.fit(df_training_set, df_training_labels)

            os.remove("model.pkl")
            os.remove("training_data.pkl")
            os.remove("training_labels.pkl")

            joblib.dump(pipeline, "model.pkl")
            joblib.dump(df_training_set, "training_data.pkl")
            joblib.dump(df_training_labels, "training_labels.pkl")

            pipeline = joblib.load("model.pkl")
        
        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))

        return jsonify("Retrained model successfully.")
    
if __name__ == '__main__':
    app.run(debug=True)