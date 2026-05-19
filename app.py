#20240802342
from flask import Flask, render_template
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load data
df = pd.read_csv("dataset/retail_data.csv")

# Preprocessing
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'] = df['Quantity'] * df['Price']

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists("static"):
        os.makedirs("static")

    # 📊 1. Bar Chart (Top Products)
    top_products = df['Product'].value_counts()

    plt.figure()
    top_products.plot(kind='bar')
    plt.title("Top Products")
    plt.xlabel("Product")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("static/bar.png")
    plt.close()

    # 📈 2. Line Chart (Sales Trend)
    sales = df.groupby('Date')['Revenue'].sum()

    plt.figure()
    sales.plot()
    plt.title("Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/line.png")
    plt.close()

    # 🥧 3. Pie Chart (Distribution)
    plt.figure()
    top_products.plot(kind='pie', autopct='%1.1f%%')
    plt.ylabel("")
    plt.title("Product Distribution")
    plt.tight_layout()
    plt.savefig("static/pie.png")
    plt.close()

    return render_template('dashboard.html')

# ---------------- RECOMMEND ----------------
@app.route('/recommend')
def recommend():
    try:
        basket = df.groupby(['CustomerID', 'Product'])['Quantity'].sum().unstack().fillna(0)

        # convert to 0/1
        basket = (basket > 0).astype(int)

        from mlxtend.frequent_patterns import apriori, association_rules

        freq_items = apriori(basket, min_support=0.1, use_colnames=True)

        if freq_items.empty:
            return "No frequent items found"

        rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)

        if rules.empty:
            return "No rules generated"

        return render_template("recommend.html", tables=rules.to_html())

    except Exception as e:
        return str(e)

# ----------------cluster ----------------
@app.route('/cluster')
def cluster():
    try:
        X = df[['Quantity', 'Price']]

        kmeans = KMeans(n_clusters=3, random_state=0)
        df['Cluster'] = kmeans.fit_predict(X)

        return render_template("cluster.html", tables=df.to_html())

    except Exception as e:
        return str(e)
# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True, port=5002)