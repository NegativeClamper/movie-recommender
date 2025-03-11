# Movie Recommender System using Scikit-Learn

This project implements a **movie recommender system** using collaborative filtering with matrix factorization techniques from **scikit-learn**. The system is trained on the **MovieLens 100K dataset** and can generate personalized movie recommendations for users, including custom input users.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Future Improvements](#future-improvements)
7. [License](#license)

---

## Introduction

The goal of this project is to build a movie recommendation system that suggests movies to users based on their preferences. The system uses **collaborative filtering** and **matrix factorization** techniques to predict user ratings and recommend movies. It is implemented using **scikit-learn** and can handle custom user input for personalized recommendations.

---

## Features

- **Collaborative Filtering**: Uses matrix factorization (Truncated SVD) to predict user ratings.
- **Custom User Input**: Allows users to input their own movie ratings and get personalized recommendations.
- **Top-N Recommendations**: Generates a list of top N recommended movies for a user.
- **Interactive**: Users can rate movies and see recommendations in real-time.

---

## Dataset

The project uses the **MovieLens 100K dataset**, which contains:
- **100,000 ratings** from 943 users on 1682 movies.
- **Movie metadata**: Titles, genres, release dates, etc.

The dataset is available in the following files:
- `u.data`: User-movie ratings.
- `u.item`: Movie metadata.

Download the dataset from [MovieLens](https://grouplens.org/datasets/movielens/100k/).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender

2. Install required libs.
3. Download the dataset

## Future Improvements
1.Improve Recommendation Quality:
  Incorporate movie genres, user demographics, and other metadata.
  Use advanced techniques like deep learning (e.g., Neural Collaborative Filtering).

2.Interactive Web Interface:
  Deploy the system as a web app using Flask or Streamlit.

3.Scalability:
   Optimize the system for larger datasets (e.g., MovieLens 1M or 10M).

4.User Feedback:
   Allow users to provide feedback on recommendations to improve the model.
## License
