from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    table = None
    graph_generated = False

    if request.method == "POST":

        stock_symbol = request.form["stock"]

        # Download stock data
        stock_data = yf.download(stock_symbol, period="1mo")

        # Reset index
        stock_data.reset_index(inplace=True)

        # Convert table to HTML
        table = stock_data.to_html(classes='table table-striped')

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
        graph_generated=graph_generated
    )

if __name__ == "__main__":
    app.run(debug=True)