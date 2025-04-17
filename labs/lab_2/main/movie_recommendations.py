import pickle
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import difflib

APP_DIR = Path(__file__).resolve().parent
LAB_DIR = APP_DIR.parent
DATA_DIR = LAB_DIR / "data"
CACHE_DIR = LAB_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Laddar och cachar CSV-filer som pickle för snabbare åtkomst
def load_or_cache_csv(name: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{name}.csv"
    pkl_path = CACHE_DIR / f"{name}.pkl"
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    df = pd.read_csv(csv_path)
    df.to_pickle(pkl_path)
    return df


# Filtrerar bort användare med för få betyg (default: minst 100)
def filter_users(ratings_df, min_ratings=100):
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    return ratings_df[ratings_df['userId'].isin(valid_users)]

# Grupperar taggar per film till en sträng
def aggregate_tags(tags_df):
    return tags_df.groupby("movieId")["tag"] \
        .agg(lambda x: " ".join(x.dropna().astype(str).unique())) \
        .reset_index()

# Slår ihop filmer med betyg och taggar till en enda DataFrame
def combine_data(movies_df, avg_ratings_df, tags_df):
    return movies_df.merge(avg_ratings_df, on="movieId", how="left") \
                    .merge(tags_df, on="movieId", how="left")

# Kombinerar genres och taggar, rensar bort dubbletter och sätter allt i lowercase
def combine_and_clean_keywords(row):
    genres = row["genres"]
    tags = row["tag"]

    if pd.isna(tags):
        tags = ""
    genres_clean = genres.replace("|", " ").lower().split()
    tags_clean = tags.lower().split()
    combined = list(set(genres_clean + tags_clean))
    return combined

# Laddar bearbetad filmdata eller skapar den om den inte finns
def load_or_create_combined_df():
    processed_path = CACHE_DIR / "processed_movies.pkl"
    keyword_path = CACHE_DIR / "keyword_data.pkl"

    if processed_path.exists() and keyword_path.exists():
        combined_df = pd.read_pickle(processed_path)
        with open(keyword_path, "rb") as f:
            vectorizer, keyword_matrix = pickle.load(f)
    else:
        filtered_ratings = filter_users(load_or_cache_csv("ratings")) \
                                .sample(n=10000, random_state=42)
        avg_ratings = filtered_ratings.groupby("movieId")["rating"] \
                                      .mean().reset_index()

        movies_df = load_or_cache_csv("movies")
        movies_df["title"] = movies_df["title"].apply(fix_title_format)

        tags_df = aggregate_tags(load_or_cache_csv("tags"))
        combined_df = combine_data(movies_df, avg_ratings, tags_df)

        combined_df["keywords"] = combined_df.apply(
            combine_and_clean_keywords, axis=1)
        combined_df.to_pickle(processed_path)

        combined_df["keywords_str"] = combined_df["keywords"] \
                                      .apply(lambda x: " ".join(x))
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        keyword_matrix = vectorizer.fit_transform(combined_df["keywords_str"])

        with open(keyword_path, "wb") as f:
            pickle.dump((vectorizer, keyword_matrix), f)

    return combined_df, keyword_matrix

# Söker efter filmer vars titel innehåller query-strängen (regex-sökning)
def search_movies_by_title(query, df):
    pattern = re.escape(query)
    return df[df["title"].str.contains(pattern, case=False, regex=True)]

# Ändrar format på titlar så som 'Matrix, The (1999)' till 'The Matrix (1999)'
# Detta behövs för att en sökning på t.ex "The Matrix" gav felaktiga träffar i det ursprungliga formatet
def fix_title_format(title):
    match = re.match(r"^(.*), (The|An|A) (\(\d{4}\))$", title)
    if match:
        name, article, year = match.groups()
        return f"{article} {name} {year}"
    return title



# Sorterar träffar efter hur nära de är query:n med fuzzy matching
def get_sorted_matches(query, matches):

    titles = matches["title"].tolist()
    sorted_titles = difflib.get_close_matches(
        query, titles, n=len(titles), cutoff=0.0)
    matches = matches.copy()
    matches["original_index"] = matches.index
    return matches.set_index("title").loc[sorted_titles].reset_index()

# Returnerar de mest lika filmerna baserat på cosine similarity på TF-IDF
def get_recommendations(index: int,
                         df: pd.DataFrame,
                         keyword_matrix: np.ndarray,
                         top_n: int = 5) -> pd.DataFrame:
    scores = cosine_similarity(keyword_matrix[index], keyword_matrix).flatten()
    df_copy = df.copy()
    df_copy["similarity"] = scores
    recs = df_copy[df_copy.index != index].dropna(subset=["rating"])
    recs = recs.sort_values(by="similarity", ascending=False)
    return recs[["title", "genres", "rating", "similarity"]].head(top_n)