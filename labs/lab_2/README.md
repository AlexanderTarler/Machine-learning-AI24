# 🎬 Filmrekommendationssystem

Detta projekt implementerar ett innehållsbaserat filmrekommendationssystem med hjälp av maskininlärningstekniker som TF-IDF och cosine similarity.

## Funktionalitet

- Sök efter en filmtitel (med fuzzy matchning).
- Kombinerar genres och taggar från MovieLens-datan.
- Räknar ut likheter mellan filmer med hjälp av TF-IDF och cosine similarity.
- Om ingen exakt titelmatchning hittas används fallback-rekommendationer baserade på nyckelord.
- Fixar titlar med format som "Matrix, The (1999)" automatiskt till "The Matrix (1999)".

## Teknik

- **TF-IDF**: Skapar en viktad nyckelordsrepresentation av varje film.
- **Cosine similarity**: Mäter hur lika filmer är varandra.
- **Streamlit**: För interaktivt webbgränssnitt.
- **Caching**: Påskyndar laddning av data och bearbetning.

## Så kör du programmet

1. Ladda hem MovieLens-datan (ml-latest)[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/).
2. Packa upp datan.
3. Placera allt utöver "genome"-filerna i `data/`:
4. Kör programmet med:

```bash
streamlit run app.py
```

5. Gå till den adress som står i den kommandotolk du använder.

6. Skriv in en filmtitel i inputfältet – exempel:
   - `The Matrix`
   - `Spider-Man`
   - `Toy Story`
