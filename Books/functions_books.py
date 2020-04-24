import pickle
import numpy as np
from fuzzywuzzy import process

## depickle
with open('Model/pickle_model_b.p', "rb") as file:
        trained_model = pickle.load(file)
with open('Model/pickle_Q_b.p', "rb") as file1:
        Q = pickle.load(file1)
with open('Model/pickle_sim_long.p', "rb") as file3:
        UU = pickle.load(file3)
with open('Model/pickle_dictionaries.p', "rb") as file2:
        dictionaries = pickle.load(file2)
reference_dict = dictionaries[0]  #id : title
reference_dict_authors = dictionaries[1] # id : authors
reference_dict_id_loc = dictionaries[2] # id: loc/index
#reference_id_loc_sim = dictionaries[3]  # id: loc/index
reference_dict_flipped = dict((v,k) for k,v in reference_dict.items())
reference_dict_loc_id = dict((v,k) for k,v in reference_dict_id_loc.items())




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
            #book_rec.append(fuzzy[0])
            # get id of book:

            book_id_sim = reference_dict_flipped[fuzzy[0]]
            #print(book_id_sim)
            # convert book_id into loc/index
            loc_sim = reference_dict_id_loc[book_id_sim]
            print(loc_sim)
            # get  2 neighbour-books:
            sim_ids = UU.loc[loc_sim].sort_values(ascending=False)[1:6]
            #print(UU.loc[book_id_sim].sort_values(ascending=False))
            sim_ids = sim_ids.to_frame()

            for i in range(5):

                title = reference_dict[sim_ids.index[i]]
                if title not in books:
                    book_rec.append(title)
                    book_rec.append(get_author(sim_ids.index[i]))


        except KeyError:
            for _ in range(5):
                book_rec.append("There should be a title here...")
                book_rec.append(" some author")

        recommendations_lol.append(book_rec)


    return recommendations_lol

# rec = similar_books(["Anna Karenina", "To kill a mockingbird"])
# rec
#
# # reference_dict_flipped['The Fault in Our Stars']
# #reference_dict[4]
# #

# loc_sim = reference_id_loc_sim[4]
# reference_id_loc_sim
