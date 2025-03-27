# mixa tags/genres i samma df och se till att de är unika, e.g, fantasy = Fantasy
# standardisera ratings
# rekommendera filmer så härmed "fuzzy search"/Jaccard-similarity:
# 1. filmer med samma genres/tags & enbart de genres/tags, sorterade efter ratings.
# 2. filmer med samma genres/tags +/- 1 genre/tag, sorterade efter ratings.
# 3. etc etc

# För att standardisera ratings så kan vi ge dem en "z-score": z = (x − μ)​ / σ

# Filtrera användare som har gett färre än 1-2 ratings för att sortera ut "reaktiva reviews"

# Använd TF-IDF på tags,då kan man mäta hur "unika" vissa ord är för en film, och hur viktiga de är när man jämför filmer

import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")


def filter_users(ratings_df, min_ratings=3):

    # visar hur många ratings en unik användare har lämnat i hela datan
    user_counts = ratings_df['userId'].value_counts()

    # filtrerar bort de användare som har färre än 3 ratings
    valid_users = user_counts[user_counts >= min_ratings].index

    # returnerar ett nytt DataFrame med bara de aktiva (3+ ratings) användarna kvar
    return ratings_df[ratings_df['userId'].isin(valid_users)]


def aggregate_tags(tags_df):
    tags_agg = tags_df.groupby("movieId")["tag"].agg(
        lambda x: " ".join(x.dropna().astype(str).unique())
    ).reset_index()
    return tags_agg


def combine_data(ratings_df, movies_df, tags_df):
    first_merge = ratings_df.merge(movies_df, on="movieId", how="inner")
    final_merge = first_merge.merge(
        tags_df, on="movieId", how="left").drop("timestamp", axis=1)
    return final_merge

# sätt ihop genres/tags via movieId


def combine_and_clean_keywords(row):
    genres = row["genres"]
    tags = row["tag"]

    # hanterar ev. NaN i tags ifall-ifall
    if pd.isna(tags):
        tags = ""

    # rensar och splittar till listor
    genres_clean = genres.replace("|", " ").lower().split()
    tags_clean = tags.lower().split()

    # kombinerar och tar bort dubbletter
    combined = list(set(genres_clean + tags_clean))

    return combined


filtered_ratings = filter_users(ratings).sample(n=10000, random_state=42)

aggregated_tags = aggregate_tags(tags)

combined_df = combine_data(filtered_ratings, movies, aggregated_tags)
combined_df["keywords"] = combined_df.apply(combine_and_clean_keywords, axis=1)

print(combined_df)
