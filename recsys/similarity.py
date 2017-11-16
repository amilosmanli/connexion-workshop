import pandas as pd
import numpy
from scipy.spatial.distance import cosine


def cosine_sim(rating1, rating2):
    return 1 - cosine(rating1, rating2)


def common_sim(rating1, rating2):
    rating1 = rating1[rating1 != 0]
    rating2 = rating2[rating2 != 0]
    return len(rating1.index.intersection(rating2.index))


def pearson_sim(rating1, rating2):
    return numpy.corrcoef(list(rating1), list(rating2))[0, 1]


def jaccard_sim(rating1, rating2):
    set_1 = set(rating1[rating1 != 0].index)
    set_2 = set(rating1[rating2 != 0].index)

    intersection_cardinality = len(set.intersection(*[set_1, set_2]))
    union_cardinality = len(set.union(*[set_1, set_2]))
    return intersection_cardinality / float(union_cardinality)


def calculate_distance(rating1, rating2, distance_metric):
    if distance_metric == 'intersection':
        return common_sim(rating1, rating2)
    if distance_metric == 'pearson':
        return pearson_sim(rating1, rating2)
    if distance_metric == 'jaccard':
        return jaccard_sim(rating1, rating2)
    if distance_metric == 'cosine':
        return cosine_sim(rating1, rating2)
    else:
        raise Exception('the metric specified is not implemented!')


def compute_nearest_neighbours(item, ratings_matrix, distance_metric):
    distances = []
    rating1 = ratings_matrix.ix[item]
    for item in ratings_matrix.index:
        distance = calculate_distance(rating1,
                                      ratings_matrix.ix[item],
                                      distance_metric)
        distances.append((item, distance))
    distances = pd.DataFrame(distances, columns=['item', 'similarity'])\
        .sort_values(by='similarity', ascending=False)
    return distances
