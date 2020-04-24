""" Model using Collaborative Filtering"""

from scipy.spatial import distance
import numpy as np
import pandas as pd
#from preprocess_data_books import matrix2



def model_book_similarities(data):

    data = data.T

    # make a matrix for similarities of books and put in cosine similarity
    sim_matrix = np.zeros((len(data), len(data)))
    sim_matrix = pd.DataFrame(sim_matrix, index=data.index, columns=data.index)

    for u in sim_matrix.index:  #takes a looooong time. for 10.000: - 1h 45 min
        for v in sim_matrix.index:
            sim_matrix.loc[u][v] = 1 - distance.cosine(data.loc[u], data.loc[v])

    #U2 = distance.squareform(distance.pdist(data, metric='cosine'))
    #UM = pd.DataFrame(U2)

    return sim_matrix

#m = model_book_similarities(matrix2)
