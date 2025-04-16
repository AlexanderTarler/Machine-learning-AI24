from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


def prepare_data_for_ml(combined_df):
    df = combined_df.copy()

    df = df[df["rating"].notna()].copy()

    df["rating_count"] = df.groupby("movieId")["rating"].transform("count")

    le = LabelEncoder()
    df["movie_encoded"] = le.fit_transform(df["movieId"])

    features = df[["movie_encoded", "rating_count"]]
    target = df["rating"]

    return train_test_split(features, target, test_size=0.2, random_state=42)


def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("\n[Utvärdering av modell]")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    return predictions
