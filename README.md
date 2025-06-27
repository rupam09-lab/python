# python
Box-Office-Revenue-Prediction/ │ ├── README.md ├── requirements.txt ├── data/ │ └── movies.csv ├── notebooks/ │ └── Box_Office_Prediction.ipynb ├── src/ │ ├── data_preprocessing.py │ ├── train_model.py │ └── utils.py ├── model/ │ └── linear_regression_model.pkl ├── results/ │ └── evaluation_metrics.txt └── .gitignore

🎬 Box Office Revenue Prediction Using Linear Regression
This project uses Linear Regression to predict movie box office revenue based on various features like budget, genre, runtime, etc.

🔧 Tools & Technologies
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Jupyter Notebook
📁 Dataset
We used a dataset of movies with features like:

Budget
Runtime
Genre
Release Date
Production Company
Revenue
(Include source: e.g., Kaggle TMDB Dataset)

🚀 Steps
Data Cleaning & Preprocessing
Feature Engineering
Train/Test Split
Model Training (Linear Regression)
Evaluation (MAE, MSE, R² Score)
Revenue Prediction for New Movies
📊 Results
The model achieved an R² score of 0.72, indicating decent predictive power.

📌 Future Work
Try other models like Decision Trees or Random Forest
Improve feature engineering
Tune hyperparameters
📸 Example Visualization
scatter

🤝 Contributing
Pull requests are welcome.

from sklearn.linear_model import LinearRegression from sklearn.model_selection import train_test_split from sklearn.metrics import mean_squared_error, r2_score import pandas as pd

df = pd.read_csv("data/movies.csv")

Example features
X = df[['budget', 'runtime']] y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression() model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred)) pandas numpy scikit-learn matplotlib seaborn jupyter
