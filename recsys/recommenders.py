import pandas as pd


def sort_recommendations(recommendations, N):
    recommendations.sort()
    recommendations.reverse()
    recommendations = recommendations[0:N]
    return pd.DataFrame(recommendations, columns=['rating', 'movie'])


def recommend_movie_with_uknn(ratings, similarity_matrix, num_of_recs):
    neighbours = similarity_matrix
    recommendations = {}
    supportRatings = {}
    threshold = 6

    for neighbour in neighbours.item.unique():

        weight = neighbours.similarity[neighbours.item == neighbour]
        neighbour_ratings = ratings.ix[ratings.customer == neighbour]

        # calculate the predicted rating for each recommendations
        for movie in neighbour_ratings.movie.unique():
            prediction = neighbour_ratings.rating[
                             neighbour_ratings.movie == movie] * weight.values[
                             0]
            # if there is a new movie, set the similarity and sums to 0
            recommendations.setdefault(movie, 0)
            supportRatings.setdefault(movie, 0)
            recommendations[movie] += prediction.values[0]
            supportRatings[movie] += 1

    # normalise so that the sum of weights for each movie adds to 1
    recs_normalized = [
        (recommendations * min((supportRatings[movie] - threshold), 1), movie)
        for
        movie, recommendations in recommendations.items()]

    return sort_recommendations(recs_normalized, num_of_recs)
