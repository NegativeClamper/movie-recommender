import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv('ml-100k/ml-100k/u.data', sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                     names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Perform matrix factorization
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_

# Calculate cosine similarity between users and items
user_similarity = cosine_similarity(user_factors)
item_similarity = cosine_similarity(item_factors.T)

# Example: Get top 10 recommendations for user 1
user_id = 1
user_ratings = user_item_matrix.loc[user_id]
predicted_ratings = np.dot(user_factors[user_id - 1], item_factors)
top_movies = np.argsort(predicted_ratings)[::-1][:10]

for movie_id in top_movies:
    print(movies[movies['movieId'] == movie_id]['title'].values[0])