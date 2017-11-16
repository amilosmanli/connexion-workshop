import os
import pandas as pd
from recsys import utils, recommenders, similarity

dataset_folder = os.path.join(os.getcwd(), 'data')
similarity_matrices_folder = os.path.join(dataset_folder, 'similarity_matrices')
ratings_folder = os.path.join(dataset_folder, 'ratings')


def create_rating_matrix():
    ratings = pd.read_csv(os.path.join(dataset_folder, 'ratings.csv'))

    # load movie metadata
    movies_file = os.path.join(dataset_folder, 'movies.csv')
    movies = pd.read_csv(movies_file)

    # lets use movie titles instead of id's to make things more human readable
    ratings = ratings.merge(movies, on='movieId')\
        .drop(['genres', 'timestamp', 'movieId'], 1)

    ratings = ratings[['userId', 'title', 'rating']]
    ratings.columns = ['customer', 'movie', 'rating']

    ratings_matrix = ratings.pivot_table(index='customer', columns='movie',
                                         values='rating', fill_value=0)
    ratings_matrix = ratings_matrix.transpose()
    ratings_matrix.to_pickle(os.path.join(dataset_folder, 'ratings_matrix'))
    return


def create_similarity_matrix(user_id, similarity_metric, K=10):
    ratings, my_customer_number = utils.merge_datasets(dataset_folder, user_id)
    ratings.to_pickle(os.path.join(ratings_folder, str(user_id)))
    ratings_matrix = ratings.pivot_table(index='customer', columns='movie',
                                         values='rating', fill_value=0)
    neighbours = similarity.compute_nearest_neighbours(user_id,
                                                       ratings_matrix,
                                                       similarity_metric)[1:K+1]
    neighbours['similarity'] = \
        neighbours['similarity'] / neighbours.similarity.sum()

    neighbours.to_pickle(os.path.join(similarity_matrices_folder, str(user_id)))
    return


def get_similarity(movie_title, similarity_metric):
    ratings_matrix = pd.read_pickle(os.path.join(dataset_folder,
                                                 'ratings_matrix'))

    return similarity.compute_nearest_neighbours(movie_title,
                                                 ratings_matrix,
                                                 similarity_metric)


def get_recommendations(user_id, num_of_movies=10):
    ratings = pd.read_pickle(os.path.join(ratings_folder, str(user_id)))
    similarity = pd.read_pickle(os.path.join(similarity_matrices_folder, str(user_id)))
    return recommenders.recommend_movie_with_uknn(ratings, similarity, num_of_movies)


if __name__ == '__main__':
    create_rating_matrix()
    create_similarity_matrix(672, 'cosine', K=100)
    movie_title = 'Star Wars: Episode VI - Return of the Jedi (1983)'
    print("Similar movies to %s" % movie_title)
    print(get_similarity(movie_title, 'cosine')[0:10])
    print("Recommendations for user %s" % 672)
    print(get_recommendations(672))
    print("Test recommendations")
