# Two book-recommender-models

![visualization](./Start.gif)

## A Website to recommend books based on three titles you put in
Content Based Filtering and Non-negative Matrix Factorization return recommendations for three titles entered by user.

Also featuring a page with the best-rated books from the dataset.

Developed in Week 10 of the Spiced-Bootcamp using Python, fuzzywuzzy, Flask, a little HTML and CSS.

![visualization](./Recommendations.gif)

## How to use:
- Clone the repository
- Install requirements: pip install -r requirements.txt
- to run: source run_server.sh
- then open a browser on the specified location

- If you are using a much newer pickle-version than 0.7.5 it is advisable to create the pickle-files new. For that, you only need to download the two files in books_data and run python main_model_books.py.

## Dataset: Goodbooks-10k
6 million ratings for 10.000 most popular books (with most ratings) -> models trained on 750.000 ratings.

## TO DO:
- ~~fix /recommender~~
- add more documentation
- ~~update design~~
- ~~fix duplicates similarity-model~~
