
import streamlit as st
from movie_recommendations import (
    load_or_create_combined_df,
    search_movies_by_title,
    get_sorted_matches,
    get_recommendations
)

# Läser in data
combined_df, keyword_matrix = load_or_create_combined_df()

# Skapar gränssnittet och intsruktioner för användaren 
st.title("🎬 Filmrekommendationer")
st.write("Skriv in en filmtitel för att få rekommendationer baserat på likhet.")

# Skapar en textinmatning för användaren
query = st.text_input("Filmtitel", "Toy Story")

# Kör filmrekommendationsprocessen
if st.button("Hitta rekommendationer"):
# Försöker hitta filmer som innehåller sökfrasen i titeln
    matches = search_movies_by_title(query, combined_df)

# Om inga matchande filmer hittas – visa varning
    if matches.empty:
        st.warning("Ingen matchande film hittades.")
# Om exakt en film hittas – välj den automatiskt
    elif len(matches) == 1:
        target_index = matches.index[0]
        st.info(f"Visar rekommendationer baserat på: **{matches.iloc[0]['title']}**")
# Om flera filmer hittas – använd fuzzy matching för att sortera dem
    else:
        sorted_matches = get_sorted_matches(query, matches)
        movie_choice = st.selectbox("Välj rätt film", sorted_matches["title"])
        target_index = sorted_matches.loc[
            sorted_matches["title"] == movie_choice, "original_index"
        ].values[0]

# Hämtar rekommendationer baserat på vald film
    recs = get_recommendations(target_index, combined_df, keyword_matrix)
    
# Visar rubrik + rekommenderade filmer i tabellform
    st.subheader(f"Rekommendationer baserat på '{combined_df.loc[target_index, 'title']}':")
    st.dataframe(recs.reset_index(drop=True))
