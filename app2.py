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

# Load the book data
df_books = pd.read_csv('onlygenres.csv', encoding="utf8")

# Define helper functions
def cosine_similarity(X, Y, weight):
    x_weighted = X.copy()
    y_weighted = Y.copy()
    x_weighted[0] *= weight
    y_weighted[0] *= weight
    return dot(x_weighted, y_weighted) / (norm(x_weighted) * norm(y_weighted))

def book_cosine(data, original_title):
    stop_words = {"a", "an", "the", "and", "or", "but", "on", "in", "with", "of", "for", "by", "to", ":", "/", "is", "about", "-", "book"}
    
    # Ensure we are working with strings for genres
    original_genres = str(data['Genre']).split(',') if pd.notna(data['Genre']) else []

    genre_count = sum(genre_weights.get(genre.strip(), 0) for genre in original_genres)  # Strip spaces from genres

    results = []
    for index, book in df_books.iterrows():
        if book['Title'] == original_title:
            continue  # Skip the original book
        
        # Ensure we are working with strings for genres
        compare_genres = str(book['Genre']).split(',') if pd.notna(book['Genre']) else []
        overlap_genres = set(original_genres) & set(compare_genres)
        overlap_genres_count = sum(genre_weights.get(genre.strip(), 0) for genre in overlap_genres)  # Strip spaces from genres

        if overlap_genres_count > 0:
            similarity = cosine_similarity([genre_count, 1], [overlap_genres_count, 1], 0.75)
            results.append((similarity, book['Title'], book['Authors'], book['Genre']))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:10]


# Define the Flask routes
@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/search_book', methods=['POST'])
def search_book():
    title_name = request.form['title_name']
    book_data = df_books[df_books['Title'].str.contains(title_name, case=False, na=False)]
    
    if not book_data.empty:
        results = book_cosine(book_data.iloc[0], title_name)
        return jsonify(results)
    else:
        return jsonify([])

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    recommendations = [title for title in df_books['Title'] if query in title.lower()]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
