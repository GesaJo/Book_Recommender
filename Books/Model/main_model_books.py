import pickle
import pandas as pd
import numpy as np
from preprocess_data_books import umb, construct_dict, matrix2
from preprocess_data_books import dict_id_location, top_25
from model_books import model_rec
from model_sim import model_book_similarities


# create and save similarity matrix
sim_matrix = model_book_similarities(matrix2).round(5)
# sim_matrix = sim_matrix.round(5)
fl32 = sim_matrix.astype('float32')
sim_array= fl32.to_numpy()
sim_tril = np.tril(sim_array)
df_tril = pd.DataFrame(sim_tril)
df_tril.to_csv("df_tril.csv", compression="zip")

#train and save NMF-model and Q
P, Q, trained = model_rec(200, umb)
with open('pickle_model_b.p', 'wb') as file:
    pickle.dump(trained, file)
with open('pickle_Q_b.p', 'wb') as file1:
    pickle.dump(Q, file1)

# save dictionaries:
reference_dict, reference_dict_authors = construct_dict("../Model/books_data/books.csv")
dictionaries = [reference_dict, reference_dict_authors, dict_id_location, top_25]

with open('pickle_dictionaries.p', 'wb') as file2:
    pickle.dump(dictionaries, file2)
