import pickle
import numpy as np
from fuzzywuzzy import process
import pandas as pd

# depickle
with open('Model/pickle_model_b.p', "rb") as file:
        trained_model = pickle.load(file)
with open('Model/pickle_Q_b.p', "rb") as file1:
        Q = pickle.load(file1)
# with open('Model/pickle_sim_long.p', "rb") as file3:
#         UU = pickle.load(file3)
with open('Model/pickle_dictionaries.p', "rb") as file2:
        dictionaries = pickle.load(file2)
reference_dict = dictionaries[0]  #id : title
reference_dict_authors = dictionaries[1] # id : authors
reference_dict_id_loc = dictionaries[2] # id: loc/index
reference_dict_flipped = dict((v,k) for k,v in reference_dict.items())
reference_dict_loc_id = dict((v,k) for k,v in reference_dict_id_loc.items())

# load sim_matrix from read_csv
UU = pd.read_csv("Model/df_tril.zip")

def get_author(book_id):
    """ Only return the first of the authors (others may be translators, editors, etc.)"""
    authors = reference_dict_authors[book_id]
    authors = [x.strip() for x in authors.split(',')]
    return authors[0]

def get_titles(ids_list):
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
    # get titles, authors and index-location of books
    locs = []
    fuzzy_titles = []
    for book in books:
        fuzzy = process.extractOne(book, reference_dict_flipped.keys())
        fuzzy_titles.append(fuzzy[0])
        #update template:
        book_id = reference_dict_flipped[fuzzy[0]]
        try:
            loc = reference_dict_id_loc[book_id]
            new[0, loc] = 5
            locs.append(loc)
        except KeyError:
            pass
        # get authors
        fuzzy_titles.append(get_author(book_id))

    #filled out template: transform by Model
    user = trained_model.transform(new)

    # dotproduct with Q
    profile = np.dot(user, Q)

    # filter out already read books
    for i in locs:
        profile[0, i] = 0

    # profile_indexed is a list with len 7196, and then sorted by score
    #first: value, second index
    profile_indexed = list(zip(profile[0], range(len(profile[0]))))
    profile_indexed.sort(key=lambda x: x[0], reverse = True)

    # get top five book-recommendations
    recommendations = []
    for t in profile_indexed[0:15]:  # t is a tuple: (score, index)
        # get id for index
        book_id = reference_dict_loc_id[t[1]]
        recommendations.append(reference_dict[book_id])
        recommendations.append(get_author(book_id))

    return recommendations, fuzzy_titles  #fuzzy_titles are input books and their authors
#recommend_books(["Twilight", "To kill a mockingbird"])



def similar_books(books):
    """recommend books with similarity of books"""
    recommendations_lol = []
    for book in books:
        try:
            book_rec = []  # first element: title of book, second: author
            fuzzy = process.extractOne(book, reference_dict_flipped.keys())
            # get id of book:
            book_id_sim = reference_dict_flipped[fuzzy[0]]
            # convert book_id into loc/index
            loc_sim = reference_dict_id_loc[book_id_sim]
            # get  2 neighbour-books:
            sim_ids = UU.loc[loc_sim].sort_values(ascending=False)[20:26]
            sim_ids_2 = UU.loc[:][f"{loc_sim}"].sort_values(ascending=False)[20:26]
            conc = pd.concat([sim_ids, sim_ids_2])
            best = conc.sort_values(ascending=False)[:4] # returns 5 best recommendations
            best = best.to_frame()

            for i in range(1,3):
                n = best.index[i]
                sim_id_real = reference_dict_loc_id[int(best.index[i])]
                title = reference_dict[sim_id_real]
                if title not in books:
                    book_rec.append(title)
                    book_rec.append(get_author(sim_id_real))

        except KeyError:
            for _ in range(5):
                t = process.extractOne(book, reference_dict_flipped.keys())
                id_no = reference_dict_flipped[t[0]]
                a = get_author(id_no)
                book_rec.append(f'Sorry I have not read "{t[0]}"')
                book_rec.append(f"{a} yet...")

        recommendations_lol.append(book_rec)

    return recommendations_lol


# r = similar_books(["Moby Dick", "Ana Karenina", "Orlando"])
# r
