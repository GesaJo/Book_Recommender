"""The traffic-centre of the application - managing routes"""
from flask import Flask, render_template, request
from functions_books import recommend_books, similar_books, top_25, get_top


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def hey_there():
    return render_template('index.html')


@app.route('/recommender')
def recommender():
    user_input = dict(request.args)
    input_books =list(user_input.values())


    recommendations, titles = recommend_books(input_books)
    recommendations_lol = get_top(similar_books(input_books))
    return render_template('recommender.html',
                            input_books=input_books,
                            recommendations=recommendations,
                            recommendations_lol= recommendations_lol,
                            titles=titles)

@app.route("/best")
def best_rated():
    return render_template("best.html",
                            top_25=top_25)
