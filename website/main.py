import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import re

def convert_to_minutes(time_str):
    if 'min' in time_str:
        return int(re.search(r'\d+', time_str).group())
    elif 'Season' in time_str:
        return int(re.search(r'\d+', time_str).group()) * 10 * 45
    else:
        return 0
    
# Load the data
df = pd.read_csv('data.csv')
df.fillna('', inplace=True)

df['text_features'] = df['name'] + ' ' + df['creator'] + ' ' + df['starring'] + ' ' + df['genres'] + ' ' + df['describle']
df['time_in_minutes'] = df['time'].apply(convert_to_minutes)

label_encoder = LabelEncoder()
df['rating_encoded'] = label_encoder.fit_transform(df['rating'])

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text_features'])

scaler = StandardScaler()
numerical_features = ['year', 'time_in_minutes', 'rating_encoded']
scaled_numerical_features = scaler.fit_transform(df[numerical_features])

import numpy as np
from scipy.sparse import hstack

X = hstack([tfidf_matrix, scaled_numerical_features])


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(X, X)

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    item_indices = [i[0] for i in sim_scores]
    return df.iloc[item_indices]


def reccomend(title):
    recommendations = get_recommendations(title)

    arr = []

    for index, row in recommendations.iterrows():
        arr.append({
            'name': row['name'],
            'creator': row['creator'],
            'starring': row['starring'],
            'genres': row['genres'],
            'describle': row['describle'],
            'time': row['time'],
            'year': row['year'],
            'rating': row['rating'],
        })

    return arr