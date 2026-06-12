import textwrap
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, send_file
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from fpdf import FPDF

app = Flask(__name__)

# Ensure required folders always exist
os.makedirs("reports", exist_ok=True)
os.makedirs("static",  exist_ok=True)


def generate_pdf(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    for line in report_text.split("\n"):
        line = line.strip()

        if line == "":
            pdf.ln(5)
            continue

        # fpdf can't wrap long lines with no spaces (e.g. "===...===")
        # so shorten any run of repeated non-space characters
        import re
        line = re.sub(r'([^\sa-zA-Z0-9])\1{9,}', lambda m: m.group(1) * 40, line)

        pdf.multi_cell(w=0, h=8, txt=line)

    pdf.output("reports/stock_report.pdf")


@app.route("/", methods=["GET", "POST"])
def index():

    table           = None
    graph_generated = False
    predicted_price = None
    recommendation  = None
    current_price   = None
    report          = None
    mae             = None
    error           = None

    if request.method == "POST":

        stock_symbol = request.form["stock"].strip().upper()

        # ── Auto-suffix: "TCS" → "TCS.NS", "TCS.NS" stays as-is ─────────
        if "." not in stock_symbol:
            stock_symbol += ".NS"

        # ── Download data ─────────────────────────────────────────────────
        stock_data = yf.download(stock_symbol, period="3mo", auto_adjust=True)

        # Empty = bad symbol
        if stock_data.empty:
            error = f"No data found for '{stock_symbol}'. Please check the symbol and try again."
            return render_template("index.html", error=error)

        # ── Flatten MultiIndex columns (yFinance quirk) ───────────────────
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]

        # ── Reset index → date becomes a regular column ───────────────────
        stock_data.reset_index(inplace=True)

        # ── Detect date column name (could be 'Date' or 'Datetime') ──────
        #    yFinance version differences cause this inconsistency
        date_col = next(
            (c for c in stock_data.columns if str(c).lower() in ('date', 'datetime')),
            stock_data.columns[0]   # fallback: first column
        )
        stock_data.rename(columns={date_col: 'Date'}, inplace=True)

        # ── Round floats for clean display ────────────────────────────────
        float_cols = stock_data.select_dtypes(include='float64').columns
        stock_data[float_cols] = stock_data[float_cols].round(2)

        # ── Build HTML table ──────────────────────────────────────────────
        table = stock_data.to_html(index=False)

        # ── Feature Engineering ───────────────────────────────────────────
        stock_data['Prediction_Day'] = np.arange(len(stock_data))
        stock_data['MA_5']           = stock_data['Close'].rolling(5).mean()
        stock_data['MA_10']          = stock_data['Close'].rolling(10).mean()
        stock_data['Daily_Return']   = stock_data['Close'].pct_change()
        stock_data.dropna(inplace=True)

        print(stock_data.head(7))

        X = stock_data[['Prediction_Day', 'MA_5', 'MA_10', 'Daily_Return']]
        y = stock_data['Close']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae    = round(float(mean_absolute_error(y_test, y_pred)), 2)

        # ── Predict next day ──────────────────────────────────────────────
        next_day_features = [[
            len(stock_data),
            float(stock_data['MA_5'].iloc[-1]),
            float(stock_data['MA_10'].iloc[-1]),
            float(stock_data['Daily_Return'].iloc[-1])
        ]]
        predicted_price = round(float(model.predict(next_day_features)[0]), 2)
        current_price   = round(float(stock_data['Close'].iloc[-1]), 2)

        # ── Recommendation (percentage-based threshold) ───────────────────
        price_diff_pct = (predicted_price - current_price) / current_price * 100
        if price_diff_pct > 1.5:
            recommendation = "BUY"
        elif price_diff_pct < -1.5:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        trend_map = {
            "BUY":  "The stock shows positive momentum based on historical data.",
            "SELL": "The stock shows negative momentum based on historical data.",
            "HOLD": "The stock is currently moving within a stable price range."
        }
        trend_analysis = trend_map[recommendation]

        # ── Generate report text ──────────────────────────────────────────
        report = textwrap.dedent(f"""
            EXECUTIVE STOCK ANALYSIS REPORT

            Company Symbol      : {stock_symbol}
            Current Market Price: Rs. {current_price}
            AI Predicted Price  : Rs. {predicted_price}
            Investment Recommendation: {recommendation}

            ===================================
            MARKET TREND ANALYSIS
            {trend_analysis}

            BUSINESS INSIGHT
            The machine learning model predicts future stock movement by analyzing
            historical market behavior including moving averages and daily returns.
            Based on the prediction and trend comparison, the system has generated
            the above investment recommendation.

            CONCLUSION
            Investors may use this analysis as an educational decision-support tool.
            Actual investment decisions should consider additional market and
            financial factors beyond this automated analysis.

            Generated Automatically By:
            AI-Powered Automated Stock Report Generation System
        """).strip()

        generate_pdf(report)

        # ── Plot graph using real Date column ─────────────────────────────
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data['Date'], stock_data['Close'], color='#2563EB', linewidth=1.8)
        plt.title(f"{stock_symbol} Closing Price Trend", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Closing Price (Rs.)", fontsize=11)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "stock_graph.png"), dpi=120)
        plt.close()

        graph_generated = True

    return render_template(
        "index.html",
        tables=table,
        graph_generated=graph_generated,
        predicted_price=predicted_price,
        recommendation=recommendation,
        current_price=current_price,
        report=report,
        mae=mae,
        error=error,
        stock_symbol=stock_symbol if request.method == "POST" else None
    )


@app.route("/download_pdf")
def download_pdf():
    return send_file("reports/stock_report.pdf", as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))