"""Functions to get recommendations"""

import pickle
import numpy as np
from fuzzywuzzy import process
import pandas as pd

# Load and prepare
with open('Model/pickle_model_b.p', "rb") as file:
        trained_model = pickle.load(file)
with open('Model/pickle_Q_b.p', "rb") as file1:
        Q = pickle.load(file1)
with open('Model/pickle_dictionaries.p', "rb") as file2:
        dictionaries = pickle.load(file2)
reference_dict = dictionaries[0]  #id : title
reference_dict_authors = dictionaries[1] # id : authors
reference_dict_id_loc = dictionaries[2] # id: loc/index
reference_dict_flipped = dict((v,k) for k,v in reference_dict.items())
reference_dict_loc_id = dict((v,k) for k,v in reference_dict_id_loc.items())
sim_matrix = pd.read_csv("Model/df_tril.zip")


def get_author(book_id):
    """ Only return the first of the authors
    (others may be translators, editors, etc.)"""

    authors = reference_dict_authors[book_id]
    authors = [x.strip() for x in authors.split(',')]
    return authors[0]


def get_titles(ids_list):
    """get list of titles from list of ids"""

    top_books = []
    for id in ids_list:
        top_books.append(reference_dict[id])
        top_books.append(get_author(id))
    return top_books

top_25 = get_titles(dictionaries[3])


def recommend_books(books):
    """recommend books NMF"""

    # create template for new input
    new = np.full((1, 7196), 3.823)

    index_location = []
    input_clean = []
    for book in books:
        fuzzy = process.extractOne(book, reference_dict_flipped.keys())
        input_clean.append(fuzzy[0])
        book_id = reference_dict_flipped[fuzzy[0]]
        try:
            loc = reference_dict_id_loc[book_id]
            new[0, loc] = 5
            index_location.append(loc)
        except KeyError:
            pass
        input_clean.append(get_author(book_id))

    #filled out template: transform by Model
    user = trained_model.transform(new)
    profile = np.dot(user, Q)

    # filter out already read books
    for i in index_location:
        profile[0, i] = 0

    profile_indexed = list(zip(profile[0], range(len(profile[0]))))
    profile_indexed.sort(key=lambda x: x[0], reverse = True)

    recommendations = []
    for t in profile_indexed[0:15]:  # t is a tuple: (score, index)
        book_id = reference_dict_loc_id[t[1]]
        if "#" not in reference_dict[book_id]:
            recommendations.append(reference_dict[book_id])
            recommendations.append(get_author(book_id))
    return recommendations, input_clean


def similar_books(books):
    """recommend books with similarity-model"""

    recommendations_lol = []
    for book in books:
        try:
            book_rec = []  # first element: title of book, second: author
            fuzzy = process.extractOne(book, reference_dict_flipped.keys())
            book_id_sim = reference_dict_flipped[fuzzy[0]]
            loc_sim = reference_dict_id_loc[book_id_sim]
            # get  2 neighbour-books:
            sim_ids = sim_matrix.loc[loc_sim].sort_values(ascending=False)[20:26]
            sim_ids_2 = sim_matrix.loc[:][f"{loc_sim}"].sort_values(ascending=False)[20:26]
            conc = pd.concat([sim_ids, sim_ids_2])
            best = conc.sort_values(ascending=False)[:7] # returns 5 best recommendations
            best = best.to_frame()

            for i in range(1,7):
                n = best.index[i]
                sim_id_real = reference_dict_loc_id[int(best.index[i])]
                title = reference_dict[sim_id_real]
                if title not in books:
                    book_rec.append({"title": title, "author": get_author(sim_id_real)})

        except KeyError:
            t = process.extractOne(book, reference_dict_flipped.keys())
            id_no = reference_dict_flipped[t[0]]
            a = get_author(id_no)
            book_rec.append({"title": f'Sorry I have not read "{t[0]}"', "author": f"{a} yet..."})

        recommendations_lol.append(book_rec)

    return recommendations_lol


def get_top(recommendations_list):
    """get 5 books from recommendation-list with a little randomness"""

    flattened = [rec for sublist in recommendations_list for rec in sublist]
    clean_list = []
    for pair in flattened:
        # print(pair)
        if "Sorry I have not read" in pair["title"]:
            pass
        elif "#" in pair["title"]:
            pass
        elif pair in clean_list:
            pass
        else:
            clean_list.append(pair)
    return clean_list
