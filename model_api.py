from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import re
import pickle
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from urllib import parse

# Flask 애플리케이션 생성 및 CORS 설정
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
cors = CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# 요청 및 추천 결과를 저장할 리스트
request_logs = []
recommendation_logs = []

# 모델 및 전처리 도구 로드
weight_path = './model/weight'  # 적절한 경로로 변경
model_dl = load_model(os.path.join(weight_path, 'model_dl.keras'))

# 전처리 도구 로드
with open(os.path.join(weight_path, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)
with open(os.path.join(weight_path, 'gender_encoder.pkl'), 'rb') as f:
    gender_encoder = pickle.load(f)
with open(os.path.join(weight_path, 'relation_encoder.pkl'), 'rb') as f:
    relation_encoder = pickle.load(f)
with open(os.path.join(weight_path, 'age_scaler.pkl'), 'rb') as f:
    age_scaler = pickle.load(f)
with open(os.path.join(weight_path, 'encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)
with open(os.path.join(weight_path, 'direction_dict.pkl'), 'rb') as f:
    direction_dict = pickle.load(f)
with open(os.path.join(weight_path, 'gps_dict.pkl'), 'rb') as f:
    loaded_gps_dict = pickle.load(f)

def preprocess_text(text):
    text = text.strip()
    text = re.sub(r'[^ㄱ-ㅎ가-힣a-zA-Z0-9\s]', '', text)
    return text

def recommend_tourist_spots(keyword, gender, companion_relationship, age, direction, top_k=10, randomize=True):
    keyword = preprocess_text(keyword)
    keyword_vector = vectorizer.transform([keyword]).toarray()

    gender_encoded = gender_encoder.transform([gender]).reshape(-1, 1)

    companion_relationship_df = pd.DataFrame([[companion_relationship]], columns=['동반인 관계'])
    companion_relationship_encoded = relation_encoder.transform(companion_relationship_df)

    age_df = pd.DataFrame([[age]], columns=['나이'])
    age_scaled = age_scaler.transform(age_df)

    input_data = [keyword_vector, gender_encoded, companion_relationship_encoded, age_scaled]
    predictions = model_dl.predict(input_data, verbose=0)[0]

    sorted_indices = predictions.argsort()[::-1]
    sorted_indices = [idx for idx in sorted_indices if direction_dict.get(encoder.inverse_transform([idx])[0]) == direction]
    top_indices = sorted_indices[:top_k]

    if randomize:
        top_20_percent = list(map(int, random.sample(list(sorted_indices[:int(0.5 * top_k)]), int(0.2 * top_k))))
        top_40_percent_1 = list(map(int, random.sample(list(sorted_indices[int(0.5 * top_k):top_k * 3]), int(0.4 * top_k))))
        top_40_percent_2 = list(map(int, random.sample(list(sorted_indices[top_k * 3:top_k * 10]), int(0.2 * top_k))))
        remaining_indices = sorted_indices[top_k * 10:]
        extra_indices = list(map(int, random.sample(list(remaining_indices), min(int(0.2 * top_k), len(remaining_indices)))))
        recommended_indices = top_20_percent + top_40_percent_1 + top_40_percent_2 + extra_indices
        recommended_indices = recommended_indices[:top_k]
    else:
        recommended_indices = top_indices

    recommendations = []
    for idx in recommended_indices:
        place_name = encoder.inverse_transform([idx])[0]
        gps_coordinates = loaded_gps_dict.get(place_name, {"GPS 위도": None, "GPS 경도": None})
        address = gps_coordinates.get("주소")
        latitude = gps_coordinates.get("GPS 위도")
        longitude = gps_coordinates.get("GPS 경도")
        if not np.isnan(latitude) and not np.isnan(longitude):
            recommendations.append({
                "name": place_name,
                "address" : address,
                "latitude": latitude,
                "longitude": longitude
            })

    return recommendations

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    print("Received data:", data)

    request_logs.append(data)
    recommendations = recommend_tourist_spots(
        parse.unquote(data.get('keyword'), 'utf8'), parse.unquote(data.get('gender'), 'utf8'), parse.unquote(data.get('companion_relationship'), 'utf8'),
        data.get('age'), data.get('direction'), data.get('top_k', 10), data.get('randomize', True)
    )

    recommendation_logs.append(recommendations)
    return jsonify(recommendations)

@app.route('/')
def home():
    return "Flask server is running!"

if __name__ == '__main__':
    app.run(debug=True)