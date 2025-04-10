import pickle  # för att spara/ladda både vectorizer och matris
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


# Skapa cache-mapp om den inte finns
CACHE_DIR = os.path.join("labs", "lab_2", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_or_cache_csv(csv_path, pkl_path):
    if os.path.exists(pkl_path):
        return pd.read_pickle(pkl_path)
    else:
        df = pd.read_csv(csv_path)
        df.to_pickle(pkl_path)
        return df


# Ladda data (med cache)


movies = load_or_cache_csv(
    "data/movies.csv", os.path.join(CACHE_DIR, "movies.pkl"))
ratings = load_or_cache_csv(
    "data/ratings.csv", os.path.join(CACHE_DIR, "ratings.pkl"))
tags = load_or_cache_csv("data/tags.csv", os.path.join(CACHE_DIR, "tags.pkl"))


def filter_users(ratings_df, min_ratings=100):
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    return ratings_df[ratings_df['userId'].isin(valid_users)]


def aggregate_tags(tags_df):
    tags_agg = tags_df.groupby("movieId")["tag"].agg(
        lambda x: " ".join(x.dropna().astype(str).unique())
    ).reset_index()
    return tags_agg


def combine_data(movies_df, avg_ratings_df, tags_df):
    movies_full = movies_df.merge(avg_ratings_df, on="movieId", how="left")
    movies_full = movies_full.merge(tags_df, on="movieId", how="left")
    return movies_full


def combine_and_clean_keywords(row):
    genres = row["genres"]
    tags = row["tag"]
    if pd.isna(tags):
        tags = ""
    genres_clean = genres.replace("|", " ").lower().split()
    tags_clean = tags.lower().split()
    return set(genres_clean + tags_clean)


def load_or_create_combined_df():
    processed_path = os.path.join(CACHE_DIR, "processed_movies.pkl")
    keyword_path = os.path.join(CACHE_DIR, "keyword_data.pkl")

    if os.path.exists(processed_path) and os.path.exists(keyword_path):
        print("Läser från cache")
        combined_df = pd.read_pickle(processed_path)
        with open(keyword_path, "rb") as f:
            vectorizer, keyword_matrix = pickle.load(f)
    else:
        print("Skapar ny kombinerad DataFrame och TF-IDF-matris...")
        filtered_ratings = filter_users(
            ratings).sample(n=10000, random_state=42)
        avg_ratings = filtered_ratings.groupby(
            "movieId")["rating"].mean().reset_index()
        aggregated_tags = aggregate_tags(tags)
        combined_df = combine_data(movies, avg_ratings, aggregated_tags)
        combined_df["keywords"] = combined_df.apply(
            combine_and_clean_keywords, axis=1)
        combined_df.to_pickle(processed_path)

        combined_df["keywords_str"] = combined_df["keywords"].apply(
            lambda x: " ".join(x))
        vectorizer = TfidfVectorizer(stop_words="english")
        keyword_matrix = vectorizer.fit_transform(combined_df["keywords_str"])

        with open(keyword_path, "wb") as f:
            pickle.dump((vectorizer, keyword_matrix), f)

    return combined_df, keyword_matrix


def search_movies_by_title(query, df):
    pattern = fr"\b{query}\b"

    matches = df[df["title"].str.contains(pattern, case=False, regex=True)]
    return matches


def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0
    return len(intersection) / len(union)


def recommend_similar_movies(title, df, keyword_matrix, top_n=5):

    matches = search_movies_by_title(title, df)

    if matches.empty:
        print("Ingen matchande film hittades.")
        return

    if len(matches) > 1:
        titles = matches["title"].tolist()

        sorted_titles = difflib.get_close_matches(
            title, titles, n=len(titles), cutoff=0.0)

        matches = matches.copy()
        matches["original_index"] = matches.index

        sorted_matches = matches.set_index(
            "title").loc[sorted_titles].reset_index()
        print(f"Flera filmer hittades för '{title}':\n")
        for i, row in sorted_matches.iterrows():
            print(f"{i + 1}: {row['title']}")
        try:
            selection = int(input("\nAnge nummer för rätt film: "))
            if 1 <= selection <= len(sorted_matches):
                target_index = sorted_matches.loc[selection -
                                                  1, "original_index"]
            else:
                print("Ogiltigt val.")
                return
        except ValueError:
            print("Felaktig input.")
            return
    else:
        target_index = matches.index[0]

    sim_path = os.path.join(CACHE_DIR, f"similarity_{target_index}.npy")
    if os.path.exists(sim_path):
        similarity_scores = np.load(sim_path)
    else:
        similarity_scores = cosine_similarity(
            keyword_matrix[target_index], keyword_matrix).flatten()
        np.save(sim_path, similarity_scores)

    df["similarity"] = similarity_scores
    recommendations = df[df.index != target_index].copy()
    recommendations = recommendations[recommendations["rating"].notna()]

    print("\nVälj hur du vill sortera rekommendationerna:")
    print("1: Endast likhet")
    print("2: Endast betyg")
    print("3: Kombinerat (likhet + normaliserat betyg)")

    try:
        strategy = int(input("Ditt val (1-3): "))
    except ValueError:
        print("Ogiltigt val, använder likhet som standard.")
        strategy = 1

    if strategy == 2:
        recommendations = recommendations.sort_values(
            by="rating", ascending=False)
    elif strategy == 3:
        min_rating = recommendations["rating"].min()
        max_rating = recommendations["rating"].max()
        recommendations["norm_rating"] = (
            recommendations["rating"] - min_rating) / (max_rating - min_rating)
        recommendations["final_score"] = 0.5 * \
            recommendations["similarity"] + 0.5 * \
            recommendations["norm_rating"]
        recommendations = recommendations.sort_values(
            by="final_score", ascending=False)
    else:
        recommendations = recommendations.sort_values(
            by="similarity", ascending=False)

    print(f"\nRekommendationer för '{df.loc[target_index, 'title']}':\n")
    cols = ["title", "genres", "rating", "similarity"]
    if strategy == 3:
        cols.append("final_score")
    print(recommendations[cols].head(top_n))


# Kör programmet
combined_df, keyword_matrix = load_or_create_combined_df()
recommend_similar_movies("Toy Story", combined_df, keyword_matrix)
