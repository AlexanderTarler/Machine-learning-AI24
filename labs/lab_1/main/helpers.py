import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


def print_basic_info(df):
    print(df.info())
    print(df.describe())


def count_cardio_cases(df):
    sick = df["cardio"].value_counts()[1]
    healthy = df["cardio"].value_counts()[0]
    print(f"Personer med hjärt- och kärlsjukdomar: {sick}")
    print(f"Personer utan hjärt- och kärlsjukdomar: {healthy}")


# Används för att räkna hur många individer det är i varje kolesterol-kategori
def chol_value_report(data, column, labels=None):
    counts = data[column].value_counts().sort_index()
    for i, count in enumerate(counts):
        label = labels[i] if labels else i
        print(f"{label}: {count}")
    return counts


def calculate_bmi(df):
    df["height_meters"] = df["height"] / 100
    df["BMI"] = df["weight"] / (df["height_meters"] ** 2)
    df.drop(columns=["height_meters"], inplace=True)
    return df


def filter_bmi(df, min_bmi=10, max_bmi=60):
    return df[(df["BMI"] >= min_bmi) & (df["BMI"] <= max_bmi)]


def categorize_bmi(df):
    BMI_ranges = [18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    labels = ["Normal weight", "Overweight",
              "Obese Class I", "Obese Class II", "Obese Class III"]
    df["BMI_category"] = pd.cut(df["BMI"], bins=BMI_ranges, labels=labels)
    return df


def filter_blood_pressure(df, ap_hi_range=(60, 300), ap_lo_range=(30, 200)):
    return df[(df["ap_hi"].between(*ap_hi_range)) & (df["ap_lo"].between(*ap_lo_range))]


def categorize_blood_pressure(ap_hi, ap_lo):
    if ap_hi > 180 or ap_lo > 120:
        return "Hypertensive Crisis"
    elif ap_hi >= 140 or ap_lo >= 90:
        return "Stage 2 Hypertension"
    elif ap_hi >= 130 or ap_lo >= 80:
        return "Stage 1 Hypertension"
    elif ap_hi >= 120 and ap_lo < 80:
        return "Elevated"
    else:
        return "Healthy"


def apply_bp_category(df):
    df["BP_category"] = df.apply(lambda row: categorize_blood_pressure(
        row["ap_hi"], row["ap_lo"]), axis=1)
    return df

# Används för att skapa de två datasetten som behövs i labben


def create_feature_sets(df):
    df = df.copy()

    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)

    df1 = df.drop(columns=["ap_hi", "ap_lo", "height",
                  "weight", "BMI"], errors="ignore")
    df2 = df.drop(columns=["BMI_category", "BP_category",
                  "height", "weight"], errors="ignore")

    df1 = pd.get_dummies(df1, columns=[
                         "BMI_category", "BP_category", "gender", "cholesterol_label"], drop_first=True)
    df2 = pd.get_dummies(
        df2, columns=["gender", "cholesterol_label"], drop_first=True)

    return df1, df2


def split_data(df):
    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    pipe = Pipeline([
        ("standardizer", StandardScaler()),
        ("normalizer", MinMaxScaler())
    ])
    X_train_scaled = pipe.fit_transform(X_train)
    X_val_scaled = pipe.transform(X_val)
    X_test_scaled = pipe.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def run_grid_search(model, param_grid, X_train, y_train, model_name, scoring="recall"):
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Bästa parametrar för {model_name}: {grid_search.best_params_}")
    print(f"Bästa validerings-score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train_all_models(models, param_grids, X_train_1, y_train_1, X_train_2, y_train_2):
    best_models = {}
    for name, model in models.items():
        print(f"\n🔍 Tränar {name} på dataset 1...")
        best_models[f"{name}_df1"] = run_grid_search(
            model(), param_grids[name], X_train_1, y_train_1, name)

        print(f"\n🔍 Tränar {name} på dataset 2...")
        best_models[f"{name}_df2"] = run_grid_search(
            model(), param_grids[name], X_train_2, y_train_2, name)
    return best_models


def evaluate_models(best_models, X_val_1, y_val_1, X_val_2, y_val_2):
    scores = {}
    for name, model in best_models.items():
        X_val = X_val_1 if "df1" in name else X_val_2
        y_val = y_val_1 if "df1" in name else y_val_2
        preds = model.predict(X_val)
        scores[name] = accuracy_score(y_val, preds)
    return pd.DataFrame(scores.items(), columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)


def extract_best_params(best_models, param_grids, verbose=True):
    results = {}
    for model_name, model in best_models.items():
        base_model_name = model_name.split('_df')[0]
        param_grid = param_grids.get(base_model_name, {})
        best_params = model.get_params()
        relevant_params = {k: v for k,
                           v in best_params.items() if k in param_grid}
        results[model_name] = relevant_params

        if verbose:
            print(f"Bästa parametrar för {model_name}: {relevant_params}")

    return results


def predict_with_models(best_models, X_val_1, X_val_2):
    val_predictions = {}
    for name, model in best_models.items():
        print(f"Gör prediction med {name}...")
        X_val = X_val_1 if "df1" in name else X_val_2
        val_predictions[name] = model.predict(X_val)
    return val_predictions


def create_voting_classifiers(best_models):
    vote_clf_df1 = VotingClassifier(
        estimators=[
            ("logreg", LogisticRegression(
                **best_models["Logistic Regression_df1"].get_params())),
            ("svc", LinearSVC(**best_models["LinearSVC_df1"].get_params())),
            ("knn", KNeighborsClassifier(
                **best_models["KNN_df1"].get_params())),
            ("rf", RandomForestClassifier(
                **best_models["Random Forest_df1"].get_params()))
        ],
        voting="hard"
    )

    vote_clf_df2 = VotingClassifier(
        estimators=[
            ("logreg", LogisticRegression(
                **best_models["Logistic Regression_df2"].get_params())),
            ("svc", LinearSVC(**best_models["LinearSVC_df2"].get_params())),
            ("knn", KNeighborsClassifier(
                **best_models["KNN_df2"].get_params())),
            ("rf", RandomForestClassifier(
                **best_models["Random Forest_df2"].get_params()))
        ],
        voting="hard"
    )

    return vote_clf_df1, vote_clf_df2


def train_voting_classifiers(vote_clf_df1, vote_clf_df2, X_train1, y_train1, X_train2, y_train2):
    print("🔍 Tränar VotingClassifier på dataset 1...")
    vote_clf_df1.fit(X_train1, y_train1)
    print("🔍 Tränar VotingClassifier på dataset 2...")
    vote_clf_df2.fit(X_train2, y_train2)
    return vote_clf_df1, vote_clf_df2


def evaluate_voting_classifiers(vote_clf_df1, vote_clf_df2, X_val1, y_val1, X_val2, y_val2):
    preds_df1 = vote_clf_df1.predict(X_val1)
    preds_df2 = vote_clf_df2.predict(X_val2)

    acc1 = accuracy_score(y_val1, preds_df1)
    acc2 = accuracy_score(y_val2, preds_df2)

    print(f"Accuracy för VotingClassifier på df1: {acc1:.4f}")
    print(f"Accuracy för VotingClassifier på df2: {acc2:.4f}")

    chosen = "df1" if acc1 > acc2 else "df2"
    print(f"Valt dataset: {chosen}")
    return chosen, (acc1, acc2)


def get_best_model_name(results_df):
    best_row = results_df.sort_values("Accuracy", ascending=False).iloc[0]
    print(f"Bästa enskilda modellen: {best_row['Model']}")
    return best_row['Model']


def optimize_threshold(model, X_train, y_train, X_val, y_val, X_test, y_test, thresholds=[0.5, 0.4, 0.3], plot=True):

    model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold = None
    best_cm = None
    best_tp = 0
    best_y_pred = None

    for t in thresholds:
        y_pred_adjusted = (y_prob > t).astype(int)
        cm = confusion_matrix(y_test, y_pred_adjusted)
        tp = cm[1, 1]

        if tp > best_tp:
            best_tp = tp
            best_threshold = t
            best_cm = cm
            best_y_pred = y_pred_adjusted

    print(f"\nBästa tröskelvärdet: {best_threshold}")
    print(classification_report(y_test, best_y_pred))

    if plot:
        disp = ConfusionMatrixDisplay(
            best_cm, display_labels=["Frisk", "Sjuk"])
        disp.plot()

    return best_threshold, best_y_pred


def plot_cardio_by_category(df, column):
    grouped = df.groupby(column)["cardio"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(grouped.index.astype(str), grouped.values, color="salmon")
    plt.xlabel(column)
    plt.ylabel("Andel sjuka")
    plt.title(f"Hjärt- och kärlsjukdom per {column}")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_histogram(data, column, bins=20, xlabel="", ylabel="", title=""):
    plt.hist(data[column], bins=bins, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_boxplot(data, column, ylabel="", title=""):
    sns.boxplot(y=data[column])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_pie_chart(values, labels, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, startangle=90,
           colors=["lightgreen", "orange", "red"])
    ax.set_title(title)
    plt.show()


def plot_dual_hist_box(data, column, hist_bins=20, xlabel="", title=""):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(data[column], bins=hist_bins, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("Antal personer")
    plt.title(f"Histogram över {title}")
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data[column])
    plt.ylabel(xlabel)
    plt.title(f"Boxplot över {title}")
    plt.tight_layout()
    plt.show()


def plot_multiple_cardio_categories(df):
    df_cardio_smoke = df.groupby("smoke")["cardio"].mean()
    df_cardio_smoke.index = df_cardio_smoke.index.astype(str)

    df_cardio_age_group = df.groupby(pd.cut(df["age_years"], bins=range(
        30, 71, 5), right=False), observed=False)["cardio"].mean()

    df["cholesterol_label"] = pd.Categorical(df["cholesterol"].map({1: "Normalt", 2: "Över normalt", 3: "Långt över normalt"}), categories=[
                                             "Normalt", "Över normalt", "Långt över normalt"], ordered=True)
    df_cardio_chol = df.groupby("cholesterol_label")["cardio"].mean()

    categories = {
        "BP_category": df.groupby("BP_category")["cardio"].mean(),
        "BMI_category": df.groupby("BMI_category", observed=False)["cardio"].mean(),
        "cholesterol_label": df_cardio_chol,
        "smoke": df_cardio_smoke,
    }

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.figure(figsize=(18, 12))
    for i, (col, series) in enumerate(categories.items(), 1):
        plt.subplot(3, 2, i)
        plt.bar(series.index.astype(str), series.values, color=colors[i-1])
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Andel sjuka", fontsize=12)
        plt.title(
            f"Hjärt- och kärlsjukdom per {col}", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.subplot(3, 2, 5)
    plt.bar(df_cardio_age_group.index.astype(str),
            df_cardio_age_group.values, color="#9467bd")
    plt.xlabel("Åldersgrupp", fontsize=12)
    plt.ylabel("Andel sjuka", fontsize=12)
    plt.title("Hjärt- och kärlsjukdom per åldersgrupp",
              fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_heatmap(correlation_matrix):

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        vmin=-1, vmax=1
    )

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Korrelation mellan features", fontsize=16, fontweight="bold")

    plt.show()


def plot_bmi_distribution(df):
    bmi_counts = df["BMI_category"].value_counts().sort_index()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    bmi_counts.plot(kind="bar", color="skyblue")
    plt.xlabel("BMI-kategori")
    plt.ylabel("Antal personer")
    plt.title("Barplot över BMI-fördelning")

    plt.subplot(1, 2, 2)
    bmi_counts.plot(kind="pie", autopct='%1.1f%%')
    plt.ylabel("")
    plt.title("Pie chart över BMI-fördelning")

    plt.tight_layout()
    plt.show()


def plot_evaluation_results(results_df, title="Modellernas Accuracy på Valideringsdatan"):
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Accuracy", y="Model", data=results_df,
                hue="Model", palette="viridis", legend=False)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("Modell", fontsize=12)
    plt.xlim(0.6, 0.8)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()
