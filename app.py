
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Netflix Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# =========================================
# NETFLIX STYLE CSS
# =========================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to bottom right,#120000,#1a0000,#2b0000);
    color:white;
}

.block-container{
    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

.title{
    font-size:48px;
    font-weight:bold;
    color:white;
}

.subtitle{
    color:#d1d1d1;
    font-size:18px;
    margin-bottom:25px;
}

.metric-card{
    background: linear-gradient(145deg,#1c1c1c,#2a0d0d);
    border-radius:20px;
    padding:25px;
    border:1px solid rgba(229,9,20,0.25);
    box-shadow:0 0 15px rgba(229,9,20,0.15);
}

.metric-title{
    color:#d1d1d1;
    font-size:18px;
}

.metric-value{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.card{
    background: linear-gradient(145deg,#1c1c1c,#2a0d0d);
    border-radius:22px;
    padding:25px;
    border:1px solid rgba(229,9,20,0.25);
    box-shadow:0 0 15px rgba(229,9,20,0.15);
    margin-bottom:25px;
}

div.stButton > button{
    width:100%;
    background:linear-gradient(90deg,#E50914,#B20710);
    color:white;
    border:none;
    border-radius:12px;
    padding:14px;
    font-weight:bold;
    font-size:16px;
}

div.stButton > button:hover{
    background:#ff1e2d;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD DATA
# =========================================

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

data = pd.merge(ratings,movies,on="movieId")

movies['genres'] = movies['genres'].fillna("")

# =========================================
# HEADER
# =========================================

st.markdown('<div class="title">🎬 Netflix Movie Recommendation Dashboard</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Pattern Mining • PageRank • BERT-style Similarity • Movie Analytics</div>',
    unsafe_allow_html=True
)

# =========================================
# METRICS
# =========================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Total Movies</div>
    <div class="metric-value">{movies.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Total Users</div>
    <div class="metric-value">{ratings.userId.nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Total Ratings</div>
    <div class="metric-value">{ratings.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    genres_count = movies['genres'].str.split('|').explode().nunique()

    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Genres</div>
    <div class="metric-value">{genres_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================================
# TRENDING + GENRES
# =========================================

left,right = st.columns(2,gap="large")

# ---------- Trending Movies ----------

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔥 Trending Movies")

    top_movies = data['title'].value_counts().head(10)

    fig = px.bar(
        x=top_movies.values,
        y=top_movies.index,
        orientation='h',
        color=top_movies.values,
        color_continuous_scale='reds'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Genre Distribution ----------

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🎭 Genre Distribution")

    genres = movies['genres'].str.split('|').explode()

    genre_counts = genres.value_counts().head(6)

    fig2 = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        hole=0.6,
        color_discrete_sequence=px.colors.sequential.Reds
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350
    )

    st.plotly_chart(fig2,use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# FREQUENT MOVIE COMBINATIONS
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🎥 Frequent Movie Combinations (Pattern Mining)")

user_movies = data.groupby('userId')['title'].apply(list)

pairs = []

for movie_list in user_movies:
    if len(movie_list) > 1:
        pairs.extend(combinations(set(movie_list),2))

pair_counts = Counter(pairs)

top_pairs = pair_counts.most_common(10)

pair_df = pd.DataFrame(
    top_pairs,
    columns=['Movie Pair','Count']
)

pair_df['Movie Pair'] = pair_df['Movie Pair'].astype(str)

fig3 = px.bar(
    pair_df,
    x='Count',
    y='Movie Pair',
    orientation='h',
    color='Count',
    color_continuous_scale='Reds'
)

fig3.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=400
)

st.plotly_chart(fig3,use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# PAGERANK SECTION
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("⭐ Top Ranked Movies using PageRank")

G = nx.Graph()

for (m1,m2),count in top_pairs:
    G.add_edge(m1,m2,weight=count)

pagerank_scores = nx.pagerank(G)

rank_df = pd.DataFrame({
    'Movie': list(pagerank_scores.keys()),
    'Score': list(pagerank_scores.values())
})

rank_df = rank_df.sort_values(
    by='Score',
    ascending=False
).head(10)

fig4 = px.bar(
    rank_df,
    x='Score',
    y='Movie',
    orientation='h',
    color='Score',
    color_continuous_scale='Reds'
)

fig4.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=400
)

st.plotly_chart(fig4,use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# BERT STYLE SIMILARITY
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🤖 Movie Similarity Analysis")

movie_list = movies['title'].unique()

selected_movie = st.selectbox(
    "Choose a Movie",
    sorted(movie_list)
)

vectorizer = TfidfVectorizer(stop_words='english')

feature_matrix = vectorizer.fit_transform(movies['genres'])

similarity = cosine_similarity(feature_matrix)

indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()

def recommend(movie_name):

    idx = indices[movie_name]

    scores = list(enumerate(similarity[idx]))

    scores = sorted(
        scores,
        key=lambda x:x[1],
        reverse=True
    )

    scores = scores[1:6]

    movie_indices = [i[0] for i in scores]

    return movies['title'].iloc[movie_indices]

if st.button("Generate Recommendations 🍿"):

    recs = recommend(selected_movie)

    st.success("Recommended Movies")

    for movie in recs:
        st.markdown(f"### 🎬 {movie}")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# DATASET TABLE
# =========================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Dataset Preview")

st.dataframe(
    movies[['title','genres']].head(20),
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================

st.markdown("""
<center style='color:#bbbbbb'>
Netflix Style Movie Recommendation System <br>
Pattern Mining • PageRank • Similarity Analysis
</center>
""", unsafe_allow_html=True)
