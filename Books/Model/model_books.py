from sklearn.decomposition import NMF

def model_rec(n_components, data):
    """ Model for book recommender -> NMF"""
    model = NMF(n_components=n_components, init='random')
    trained = model.fit(data)
    Q = model.components_  # book-genre matrix  weights
    P = model.transform(data)  # user-genre matrix

    return P, Q, trained
