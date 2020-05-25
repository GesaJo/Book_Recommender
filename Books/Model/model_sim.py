from scipy.spatial import distance
import pandas as pd
from preprocess_data_books import matrix2


def model_book_similarities(data):
    """ Model using Collaborative Filtering"""

    data = data.T
    U2 = distance.squareform(distance.pdist(data, metric='cosine'))
    sim_matrix = pd.DataFrame(U2)

    return sim_matrix
