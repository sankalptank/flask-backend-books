from flask import Flask, render_template, request, jsonify
import pandas as pd
import csv
from numpy import dot
from numpy.linalg import norm
from collections import Counter

# Initialize Flask app
app = Flask(__name__)

# Load genre weights
genre_weights = {}
with open('genre_counts.csv', encoding="utf8", mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # skip header
    for row in csv_reader:
        genre, count, weight = row[0], int(row[1]), float(row[2])
        genre_weights[genre] = weight  # store weights of genres

# Load the author data
df_authors = pd.read_csv('author_genres_summary.csv', encoding="utf8")
author_genres = df_authors['Authors'].astype(str).tolist()

# Define helper functions
def cosine_similarity(X, Y, weight):
    x_weighted = X.copy()
    y_weighted = Y.copy()
    x_weighted[0] *= weight
    y_weighted[0] *= weight
    return dot(x_weighted, y_weighted) / (norm(x_weighted) * norm(y_weighted))

def get_author_index(dataframe, author_name):
    matching_indices = dataframe.index[dataframe['Authors'].str.contains(author_name, case=False, na=False)]
    if not matching_indices.empty:
        return dataframe.loc[matching_indices[0]]
    else:
        return None

def normalize_name(name):
    return name.strip().lower()

def author_cosine(data, original_author_name):
    original_genres = data['Genres'].split(',')
    genre_count = len(original_genres)
    counter1 = Counter(original_genres)
    results = []

    for _, author in df_authors.iterrows():
        author_name_normalized = normalize_name(author['Authors'])
        if author_name_normalized == normalize_name(original_author_name):
            continue

        genres = str(author['Genres']) if pd.notna(author['Genres']) else ''
        author_genres_list = genres.split(',')
        total_genres = len(author_genres_list)

        if total_genres == 0:
            continue

        counter2 = Counter(author_genres_list)
        overlap = list((counter1 & counter2).elements())
        overlap_genres_count = sum(genre_weights.get(genre, 0) for genre in overlap)

        if overlap_genres_count == 0:
            continue

        normalized_genre_count = sum(genre_weights.get(genre, 0) for genre in original_genres) / genre_count
        normalized_overlap_count = overlap_genres_count / total_genres

        similarity = cosine_similarity([normalized_genre_count, 0], [normalized_overlap_count, 1], 0.25)
        
        if similarity > 0.10 and int(author['Number Of Books']) > 10:
            results.append((similarity, author['Authors'], author['Number Of Books']))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:10]

# Define the Flask routes
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/search_author', methods=['POST'])
def search_author():
    author_name = request.form['author_name']
    author_data = get_author_index(df_authors, author_name)
    if author_data is not None:
        results = author_cosine(author_data, author_name)
        return jsonify(results)
    else:
        return jsonify([])

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    recommendations = [author for author in author_genres if query in author.lower()]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
