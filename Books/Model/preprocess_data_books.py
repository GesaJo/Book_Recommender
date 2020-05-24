"""Loading and preprocessing the data"""
import pandas as pd
import numpy as np

# preprocessing data for NMF
df = pd.read_csv('../Model/books_data/ratings_sm.csv')
df_large = df[:750000]
user_matrix = pd.pivot_table(df_large, values='rating', index='user_id', columns='book_id')
umb = user_matrix.fillna(user_matrix.mean().mean())

# what colum-name corresponds to location of column
dict_id_location = {}
for num, i in enumerate(umb.columns):
    dict_id_location[i] = num

#smaller dataframe for similarity-model
df_picked = pd.read_csv("../Model/books_data/books_data_picked.csv")
df_small = df_picked[:750000]
um_small = pd.pivot_table(df_small, values='rating', index='user_id', columns='book_id')
matrix2 = um_small.fillna(um_small.mean().mean())

#top25 rated books
mean_rating = df.groupby('book_id').mean()
top_25 = mean_rating.sort_values('rating', ascending=False)[:25].index


def construct_dict(path):
    df_titles = pd.read_csv(path)
    d1 = df_titles[['book_id','authors', 'original_publication_year', 'original_title',
            'title', 'language_code', 'isbn', 'isbn13', 'image_url',
            'small_image_url', 'average_rating','ratings_count',
            'work_ratings_count', 'work_text_reviews_count', 'ratings_1',
            'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
            'goodreads_book_id', 'best_book_id', 'work_id', 'books_count']]
    d2 = d1.iloc[:,0:12]

    reference_dict_b = pd.Series(d2["title"].values,index=d2["book_id"]).to_dict()
    reference_dict_authors = pd.Series(d2["authors"].values, index=d2["book_id"]).to_dict()

    return reference_dict_b, reference_dict_authors
