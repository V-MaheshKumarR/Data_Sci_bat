import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="BookMind Recommender", layout="wide")

# =====================================================
# Synthetic Dataset
# =====================================================
@st.cache_data
def clean_data():
    books=pd.read_csv(r"D:\Data Science_bat\Recommendation_system\data\Books.csv")
    ratings=pd.read_csv(r"D:\Data Science_bat\Recommendation_system\data\Ratings.csv")
    users=pd.read_csv(r"D:\Data Science_bat\Recommendation_system\data\Users.csv")
    books.columns = [c.strip() for c in books.columns]
    books = books.rename(columns={
        "Book-Title": "title",
        "Book-Author": "author",
        "Year-Of-Publication": "year",
        "Publisher": "publisher",
    })

    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    books = books.dropna(subset=["ISBN", "title"]).drop_duplicates("ISBN")

    ratings = ratings.rename(columns={
        "User-ID": "user_id",
        "Book-Rating": "rating",
    })
    ratings = ratings.dropna()
    ratings = ratings[ratings["rating"] > 0]

    users = users.rename(columns={"User-ID": "user_id"})
    users = users.dropna(subset=["user_id"])

    return books, ratings, users


# =====================================================
# Build Recommendation System
# =====================================================
@st.cache_data
def build_system():
    books, ratings, users = clean_data()

    merged = ratings.merge(books[['ISBN','title']], on='ISBN')
    pivot = merged.pivot_table(index='title', columns='user_id', values='rating').fillna(0)

    similarity = cosine_similarity(pivot)

    # KMeans clustering (user personas)
    user_matrix = pivot.T
    scaler = StandardScaler()
    X = scaler.fit_transform(user_matrix)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    user_cluster = dict(zip(user_matrix.index, cluster_labels))

    return books, ratings, pivot, similarity, user_matrix, user_cluster, cluster_labels


books, ratings, pivot, similarity, user_matrix, user_cluster, cluster_labels = build_system()

# =====================================================
# USER-BASED RECOMMENDATION FUNCTION
# =====================================================
def recommend_for_user(user_id, top_n=5):

    if user_id not in user_matrix.index:
        return None, "User not found."

    cluster_id = user_cluster[user_id]

    # Find peers in same cluster
    peers = [u for u, c in user_cluster.items() if c == cluster_id and u != user_id]

    if len(peers) == 0:
        return None, "No similar users found."

    target_vector = user_matrix.loc[user_id].values.reshape(1, -1)
    peer_matrix = user_matrix.loc[peers]

    similarities = cosine_similarity(target_vector, peer_matrix)[0]

    read_books = set(user_matrix.columns[user_matrix.loc[user_id] > 0])
    unread_books = [b for b in user_matrix.columns if b not in read_books]

    scores = []

    for book in unread_books:
        peer_ratings = peer_matrix[book].values
        mask = peer_ratings > 0

        if mask.sum() == 0:
            continue

        weighted_score = np.dot(similarities[mask], peer_ratings[mask]) / (
            np.abs(similarities[mask]).sum() + 1e-9
        )

        scores.append((book, weighted_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_n], None


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("📚 BookMind — Hybrid Recommendation System")

tab1, tab2 = st.tabs(["🔍 Similar Books", "👤 User Recommendations"])

# -----------------------------------------------------
# ITEM-BASED (Book-to-Book)
# -----------------------------------------------------
with tab1:
    st.subheader("Find Similar Books")

    book_list = pivot.index.tolist()
    selected_book = st.selectbox("Select a Book", book_list)

    top_n_item = st.slider("Number of recommendations", 3, 10, 5, key="item_slider")

    if st.button("Recommend Similar Books"):
        idx = book_list.index(selected_book)

        scores = sorted(
            enumerate(similarity[idx]),
            key=lambda x: x[1],
            reverse=True
        )[1:top_n_item+1]

        st.success("Top Similar Books:")
        for i, score in scores:
            st.write(f"📖 {book_list[i]}  —  {round(score*100,1)}% similarity")


# -----------------------------------------------------
# USER-BASED (Cluster + CF Hybrid)
# -----------------------------------------------------
with tab2:
    st.subheader("Personalized User Recommendations")

    user_id = st.number_input(
        "Enter User ID",
        min_value=int(user_matrix.index.min()),
        max_value=int(user_matrix.index.max()),
        value=1
    )

    top_n_user = st.slider("Number of recommendations", 3, 10, 5, key="user_slider")

    if st.button("Get Recommendations"):
        recommendations, error = recommend_for_user(user_id, top_n_user)

        if error:
            st.error(error)
        else:
            cluster_id = user_cluster[user_id]
            st.success(f"User belongs to Cluster {cluster_id}")

            st.markdown("### Recommended Books:")
            for book, score in recommendations:
                st.write(f"📘 {book}  —  Predicted Score: {round(score,2)}")


# -----------------------------------------------------
# Cluster Distribution
# -----------------------------------------------------
st.markdown("---")
st.subheader("👥 User Cluster Distribution")

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
st.bar_chart(cluster_counts)