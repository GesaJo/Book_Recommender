import pickle
from preprocess_data_books import umb, construct_dict, matrix2
from preprocess_data_books import dict_id_location, top_25
from model_books import model_rec
from model_sim import model_book_similarities

P, Q, trained = model_rec(200, umb)
sim_matrix = model_book_similarities(matrix2)

# save similarity_model
with open('pickle_sim_long.p', 'wb') as file3:
    pickle.dump(sim_matrix, file3)


# save trained model:
with open('pickle_model_b.p', 'wb') as file:
    pickle.dump(trained, file)

# save Q
with open('pickle_Q_b.p', 'wb') as file1:
    pickle.dump(Q, file1)

# save dictionaries:
reference_dict, reference_dict_authors = construct_dict("../Model/books_data/books.csv")
dictionaries = [reference_dict, reference_dict_authors, dict_id_location, top_25]

with open('pickle_dictionaries.p', 'wb') as file2:
    pickle.dump(dictionaries, file2)
