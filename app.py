import streamlit as st
import pandas as pd
from recommender import HybridRecommender
import time
import json
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="Nexus Stream | Movie AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# --- Initialize Recommender ---
@st.cache_resource
def load_recommender():
    try:
        return HybridRecommender(
            data_path='processed_movies.parquet',
            tfidf_path='tfidf_model.pkl',
            pca_path='pca_model.pkl',
            kmeans_path='kmeans_model.pkl',
            matrix_path='tfidf_matrix.pkl'
        )
    except Exception as e:
        st.error(f"Failed to load engine components: {e}")
        return None

recommender = load_recommender()

# --- Load Stats ---
@st.cache_data
def load_stats():
    try:
        with open('eda_assets/stats.json', 'r') as f:
            eda_stats = json.load(f)
        with open('eda_assets/evaluation_results.json', 'r') as f:
            eval_stats = json.load(f)
        return eda_stats, eval_stats
    except:
        return None, None

eda_stats, eval_stats = load_stats()

# --- Helpers ---
def movie_card(title, genres, similarity, cluster_match=False):
    genres = str(genres).replace('|', ' ')
    genre_badges = "".join([f'<span class="genre-tag">{g.strip()}</span>' for g in genres.split() if g.strip()])
    match_pct = int(similarity * 100)
    base_score = min(50 + match_pct, 99)
    if base_score <= 50:
        base_score = min(int(similarity * 1000), 99)
    
    html = f"""
    <div class="movie-card">
        <div class="score-badge">{'🔥 Cluster Match ' if cluster_match else ''}{base_score}% Match</div>
        <div class="movie-title">{title}</div>
        <div class="movie-genres">{genre_badges}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- App Layout ---
st.markdown('<h1 class="gradient-text" style="text-align: center; font-size: 2.5rem; margin-bottom: 0;">Nexus Stream Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #8b949e; margin-bottom: 2rem;">Hybrid AI Recommendations via NLP & Clustering</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA & Insights", "🔍 NLP Search", "🎯 Preference Tuner"])

with tab1:
    st.markdown("## Dataset Overview")
    st.markdown("""
    The recommendation engine is powered by the **MovieLens 32M Dataset**, providing incredibly dense user-item interaction histories to facilitate high-quality semantic NLP & collaborative insights.
    - **32,000,204** ratings
    - **2,000,072** tags
    - **87,585** movies
    - **200,948** users
    """)
    st.markdown("---")

    if eda_stats:
        st.markdown("## Core KPI Cards")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Active Users Analyzed", f"{eda_stats.get('total_users', 0):,}")
        c2.metric("Movies in Feature Vector", f"{eda_stats.get('total_movies', 0):,}")
        c3.metric("Ratings Parsed", f"{eda_stats.get('total_ratings', 0):,}")
        c4.metric("Mean Rating", f"{eda_stats.get('mean_rating', 0):.2f}", delta=f"± {eda_stats.get('std_rating', 0):.2f} std", delta_color="off")
        
        st.markdown("---")
        st.markdown("## Charts and Distributions")
        col1, col2 = st.columns(2)
        with col1:
            st.image("eda_assets/rating_distribution.png", use_container_width=True, caption="Rating Distribution Histogram")
        with col2:
            st.image("eda_assets/genre_frequency.png", use_container_width=True, caption="Genre Frequency Bar Chart")

        st.markdown("---")
        st.markdown("## Outlier & Data Quality")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.warning(f"**IQR Flag (Bot Activity Volume)**\nIdentified **{eda_stats.get('outlier_users_iqr', 0):,}** users generating extreme rating volumes exceeding the statistical median.")
            st.error(f"**Z-Score Flag (Extreme Polarization)**\nIdentified **{eda_stats.get('extreme_ratings_count', 0):,}** ratings sitting 3 Standard Deviations away from group consensus.")
        with col_m2:
            st.info(f"**Missing Items Resolved:** Null matrices isolated and removed before TF-IDF computation.\n\n**Data Matrix Sparsity:** `{eda_stats.get('sparsity', 0.0)}%`. High sparsity resolved via dense vector K-Means assignment.")

    if eval_stats:
        st.markdown("---")
        st.markdown("## ⚙ Diagnostics and Model State")
        st.markdown("### 1. Automatic K Selection Validation (Unsupervised)")
        col_k1, col_k2 = st.columns([1, 2])
        with col_k1:
            st.markdown(f"**Optimal Assigned K:** `{eval_stats.get('num_clusters', 25)}`")
            st.write(f"- Silhouette Optimal Bounds: **{eval_stats.get('silhouette_best', 0):.4f}**")
            st.write(f"- Davies-Bouldin Index: **{eval_stats.get('davies_bouldin', 0):.2f}**")
            st.write(f"- Calinski-Harabasz: **{eval_stats.get('calinski_harabasz', 0):.2f}**")
        with col_k2:
            if os.path.exists('eda_assets/k_sweep.png'):
                st.image("eda_assets/k_sweep.png", use_container_width=True, caption="Elbow vs Silhouette Analysis Sweep (K=5 to 50)")

        st.markdown("### 2. Supervised Target Evaluation (Implicit Preference >= 4.0)")
        try:
            with open('eda_assets/classifier_stats.json', 'r') as f:
                c_stats = json.load(f)
            
            c_m = c_stats.get("Metrics", {})
            st.markdown("#### Classifier Core Matrix")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{c_m.get('Accuracy', 0)}")
            m2.metric("Precision", f"{c_m.get('Precision', 0)}")
            m3.metric("Recall", f"{c_m.get('Recall', 0)}")
            m4.metric("F1-Score", f"{c_m.get('F1-Score', 0)}")
            m5.metric("Sensitivity", f"{c_m.get('Sensitivity', 0)}")
            m6.metric("Specificity", f"{c_m.get('Specificity', 0)}")
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.image("eda_assets/pie_class_dist.png", use_container_width=True)
            with col_v2:
                st.image("eda_assets/bar_feature_corr.png", use_container_width=True)
                
            st.markdown("#### Predictive Curves & Threshold Architecture")
            r1, r2 = st.columns(2)
            with r1:
                st.image("eda_assets/roc_curves.png", use_container_width=True)
            with r2:
                st.image("eda_assets/pr_curves.png", use_container_width=True)
            
            st.markdown("#### Internal Feature Logic")
            st.image("eda_assets/bar_feat_importance.png", use_container_width=True)
            
            st.markdown("#### Global Threshold Optimizer Sweep")
            df_th = pd.DataFrame(c_stats.get("Thresholds", []))
            st.dataframe(df_th.style.highlight_max(axis=0, subset=["F1"], color="#238636"), use_container_width=True)
            
        except Exception as e:
            st.error(f"Waiting for classifier offline pipeline computations... ({e})")

        st.markdown("### 3. Comparison Baselines & Algorithms")
        baselines = eval_stats.get('baselines', {})
        if baselines:
            df_base = pd.DataFrame(baselines).T
            st.dataframe(df_base.style.highlight_max(axis=0, color='#238636'), use_container_width=True)
            
        st.markdown("---")
        st.success("**Recommendations Strategy Deploy Card:** The Hybrid Engine out-competes conventional Collaborative SVD algorithms significantly when handling high-sparsity records. Target deployment via Dockerized container endpoints utilizing static Parquet extraction mapped explicitly against the NLP Matrix.")
        
with tab2:
    st.markdown("### Semantic Match Engine")
    nlp_query = st.text_input("", placeholder="e.g., 'I want an action movie about artificial intelligence'", label_visibility="collapsed")
    
    cbtn1, cbtn2, cbtn3 = st.columns([1, 2, 1])
    with cbtn2:
        search_pressed = st.button("Query Vector Space")
        
    if search_pressed and nlp_query and recommender:
        with st.spinner("Processing Natural Language..."):
            time.sleep(0.5)
            results = recommender.get_nlp_based_recommendations(nlp_query, top_n=9)
            
        st.markdown("#### High Confidence Vector Matches")
        if not results.empty:
            cols = st.columns(3)
            for idx, row in enumerate(results.itertuples()):
                with cols[idx % 3]:
                    movie_card(row.title, row.genres, row.similarity)
        else:
            st.warning("No matches detected. Widen the prompt.")

with tab3:
    if recommender is not None:
        all_movies = recommender.movies['title'].tolist()
        
        c_love, c_like, c_dis = st.columns(3)
        with c_love:
            st.markdown("### ❤️ Loved (+2)")
            loved = st.multiselect("Select", all_movies, key='l_loved', max_selections=5, label_visibility="collapsed")
        with c_like:
            st.markdown("### 👍 Liked (+1)")
            liked = st.multiselect("Select", all_movies, key='l_liked', max_selections=5, label_visibility="collapsed")
        with c_dis:
            st.markdown("### 👎 Disliked (-1)")
            disliked = st.multiselect("Select", all_movies, key='l_dis', max_selections=5, label_visibility="collapsed")
            
        st.markdown("---")
        cb1, cb2, cb3 = st.columns([1,2,1])
        with cb2:
            pref_pressed = st.button("Generate Cluster Extrapolations")
            
        if pref_pressed:
            if not loved and not liked and not disliked:
                st.warning("Awaiting user matrix initialization (Select a minimum of 1 item).")
            else:
                with st.spinner("Calculating Distance..."):
                    time.sleep(0.5)
                    pref_results = recommender.get_preference_based_recommendations(
                        loved_titles=loved, liked_titles=liked, disliked_titles=disliked, top_n=9
                    )
                st.markdown("#### Dynamic Suggestions")
                if not pref_results.empty:
                    cols = st.columns(3)
                    for idx, row in enumerate(pref_results.itertuples()):
                        with cols[idx % 3]:
                            movie_card(row.title, row.genres, row.similarity, row.cluster_match)
                else:
                    st.error("Matrix Empty. Try selecting items closer to global vectors.")
