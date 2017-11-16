import os
import logging
import time
import pandas as pd


def load_personal_ratings(datasets_folder, ratings_file, customer_number):
    # load personal similarity_matrices and format into the right format
    my_ratings_file = os.path.join(datasets_folder, ratings_file)
    my_ratings = pd.read_csv(my_ratings_file)
    my_ratings['userId'] = customer_number
    my_ratings['timestamp'] = int(round(time.time() * 1000))
    my_ratings = my_ratings[['userId', 'movieId', 'rating', 'timestamp']]
    logging.info("loaded %d personal similarity_matrices", len(my_ratings.index))

    my_ratings.to_csv(os.path.join(datasets_folder,
                                   'user_%s_ratings.csv' % customer_number))

    return my_ratings


def merge_datasets(dataset_folder, customer_number):
    ratings_file = os.path.join(dataset_folder, 'ratings.csv')
    ratings = pd.read_csv(ratings_file)
    my_ratings_file = os.path.join(dataset_folder,
                                   'user_%s_ratings.csv' % customer_number)
    my_ratings = pd.read_csv(my_ratings_file)
    ratings = ratings.append(my_ratings)

    # load movie metadata
    movies_file = os.path.join(dataset_folder, 'movies.csv')
    movies = pd.read_csv(movies_file)

    # lets use movie titles instead of id's to make things more human readable
    ratings = ratings.merge(movies, on='movieId')\
        .drop(['genres', 'timestamp', 'movieId'], 1)

    ratings = ratings[['userId', 'title', 'rating']]
    ratings.columns = ['customer', 'movie', 'rating']

    return ratings, customer_number
