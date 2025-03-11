import pandas as pd
import numpy as np

# Load data
ratings = pd.read_csv('ml-100k/ml-100k/u.data', sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                     names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

from sklearn.decomposition import TruncatedSVD

# Perform matrix factorization
n_components = 50  # Number of latent factors
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)  # User latent factors
item_factors = svd.components_  # Item latent factors

# Function to get top N recommendations for a user
def get_top_n_recommendations(user_id, user_factors, item_factors, movies, n=10):
    # Predict ratings for all movies
    predicted_ratings = np.dot(user_factors[user_id - 1], item_factors)
    
    # Get top N movie IDs
    top_movie_ids = np.argsort(predicted_ratings)[::-1][:n]
    
    # Get movie titles
    top_movies = [movies[movies['movieId'] == movie_id]['title'].values[0] for movie_id in top_movie_ids]
    return top_movies

# Example: Get top 10 recommendations for user 1
user_id = 1
top_movies = get_top_n_recommendations(user_id, user_factors, item_factors, movies, n=10)
print(f"Top 10 Recommendations for User {user_id}:")
for movie in top_movies:
    print(movie)

# Function to accept custom user ratings
def get_custom_user_ratings(movies):
    custom_ratings = {}
    print("Enter your ratings for the following movies (1-5). Press Enter to skip a movie.")
    for _, row in movies.sample(10).iterrows():  # Show 10 random movies for rating
        movie_id = row['movieId']
        title = row['title']
        rating = input(f"Rate '{title}' (1-5): ")
        if rating:
            custom_ratings[movie_id] = float(rating)
    return custom_ratings

# Add custom user ratings to the user-item matrix
custom_ratings = get_custom_user_ratings(movies)
if custom_ratings:
    # Create a new row for the custom user
    custom_user_row = np.zeros(user_item_matrix.shape[1])
    for movie_id, rating in custom_ratings.items():
        custom_user_row[movie_id - 1] = rating  # Movie IDs start from 1
    
    # Add the custom user to the user-item matrix
    user_item_matrix.loc[0] = custom_user_row  # Use 0 as the custom user ID

    # Retrain the model with the updated user-item matrix
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_

    # Generate recommendations for the custom user
    custom_user_id = 0  # Use 0 as the custom user ID
    top_movies = get_top_n_recommendations(custom_user_id, user_factors, item_factors, movies, n=10)
    print("\nTop 10 Recommendations for You:")
    for movie in top_movies:
        print(movie)