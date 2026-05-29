from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    table = None
    graph_generated = False
    predicted_price = None

    if request.method == "POST":

        stock_symbol = request.form["stock"]

        # Download stock data
        stock_data = yf.download(stock_symbol, period="1mo")

        # Reset index
        stock_data.reset_index(inplace=True)

        # Convert table to HTML
        table = stock_data.to_html(classes='table table-striped')

        # Machine Learning Prediction

        stock_data['Prediction_Day'] = np.arange(len(stock_data))

        X = stock_data[['Prediction_Day']]

        y = stock_data['Close']

        model = LinearRegression()

        model.fit(X, y)

        next_day = [[len(stock_data)]]

        predicted_price = model.predict(next_day)

        predicted_price = round(float(predicted_price[0]), 2)

        # Create stock graph
        plt.figure(figsize=(10,5))

        plt.plot(stock_data.index, stock_data['Close'])

        plt.title(f"{stock_symbol} Closing Price Trend")

        plt.xlabel("Date")

        plt.ylabel("Closing Price")

        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save graph image
        graph_path = os.path.join("static", "stock_graph.png")

        plt.savefig(graph_path)

        plt.close()

        graph_generated = True

    return render_template(
      "index.html",
      tables=table,
      graph_generated=graph_generated,
      predicted_price=predicted_price
    )

if __name__ == "__main__":
    app.run(debug=True)