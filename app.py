from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    table = None

    if request.method == "POST":

        stock_symbol = request.form["stock"]

        stock_data = yf.download(stock_symbol, period="1mo")

        stock_data = stock_data.reset_index()

        table = stock_data.to_html(classes='table table-striped')

    return render_template("index.html", tables=table)

if __name__ == "__main__":
    app.run(debug=True)