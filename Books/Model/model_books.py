""" Model for Book recommender -> NMF"""
from sklearn.decomposition import NMF
import numpy as np

def model_rec(n_components, data):
    model = NMF(n_components=n_components, init='random')
    trained = model.fit(data)
    Q = model.components_  # book-genre matrix  weights
    P = model.transform(data)  # user-genre matrix
    #nR = np.dot(P, Q) # The reconstructed matrix

    return P, Q, trained


#from preprocess_data_books import umb
#model_rec(5, umb)
