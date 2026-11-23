import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Sistem Rekomendasi Buku",
    page_icon="📚",
    layout="wide"
)


@st.cache_data(show_spinner="Memuat dan memproses dataset buku...")
def load_and_preprocess_data(books_source, ratings_source):
    """
    Memuat dataset Book-Crossing, membersihkan data, dan membangun
    matriks kemiripan antar buku (Item-Based Collaborative Filtering).
    """
    # 1. LOAD DATASET (separator semicolon & encoding latin-1)
    books = pd.read_csv(
        books_source,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False
    )
    ratings = pd.read_csv(
        ratings_source,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False
    )

    # Rapikan kolom yang dibutuhkan
    books = books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]]
    ratings = ratings[["User-ID", "ISBN", "Book-Rating"]]

    # 2. DATA PREPROCESSING & FILTERING
    df = ratings.merge(books, on="ISBN")

    # Filter 1: Ambil hanya rating eksplisit (rating > 0)
    df_explicit = df[df["Book-Rating"] > 0].copy()

    # Filter 2: Pengguna aktif (minimal 50 rating)
    user_counts = df_explicit["User-ID"].value_counts()
    active_users = user_counts[user_counts >= 50].index
    df_filtered = df_explicit[df_explicit["User-ID"].isin(active_users)]

    # Filter 3: Buku populer (minimal 10 rating dari pengguna aktif)
    book_counts = df_filtered["Book-Title"].value_counts()
    popular_books = book_counts[book_counts >= 10].index
    df_final = df_filtered[df_filtered["Book-Title"].isin(popular_books)]

    # 3. COLLABORATIVE FILTERING (ITEM-BASED)
    pivot_matrix = df_final.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating"
    ).fillna(0)

    item_similarity = cosine_similarity(pivot_matrix)
    similarity_df = pd.DataFrame(
        item_similarity,
        index=pivot_matrix.index,
        columns=pivot_matrix.index
    )

    # Metadata buku unik untuk memperkaya tampilan rekomendasi
    book_meta = (
        books.drop_duplicates(subset=["Book-Title"])
        .set_index("Book-Title")[["Book-Author", "Year-Of-Publication", "Publisher"]]
    )

    return df_explicit, df_final, similarity_df, book_meta


def get_popular_recommendations(df_explicit, book_meta, min_ratings=50, top_n=10):
    """
    Mengembalikan daftar buku terpopuler berdasarkan rata-rata rating tertinggi
    dengan ambang batas jumlah rating minimal.
    """
    stats = (
        df_explicit.groupby("Book-Title")
        .agg(
            num_ratings=("Book-Rating", "count"),
            avg_rating=("Book-Rating", "mean")
        )
        .reset_index()
    )

    popular_df = (
        stats[stats["num_ratings"] >= min_ratings]
        .sort_values(by=["avg_rating", "num_ratings"], ascending=[False, False])
        .head(top_n)
    )

    popular_df = popular_df.merge(book_meta, on="Book-Title", how="left")
    popular_df["avg_rating"] = popular_df["avg_rating"].round(2)

    popular_df = popular_df.rename(columns={
        "Book-Title": "Judul Buku",
        "Book-Author": "Penulis",
        "Year-Of-Publication": "Tahun Terbit",
        "Publisher": "Penerbit",
        "num_ratings": "Jumlah Rating",
        "avg_rating": "Rata-Rata Rating"
    })
    return popular_df


def recommend_similar_books(book_title, similarity_df, book_meta, top_n=5):
    """
    Mengembalikan daftar buku yang paling mirip dengan judul buku yang dipilih
    menggunakan skor Cosine Similarity.
    """
    if book_title not in similarity_df.index:
        return None

    similar_scores = similarity_df[book_title].sort_values(ascending=False)
    recommendations = similar_scores.iloc[1 : top_n + 1].reset_index()
    recommendations.columns = ["Judul Buku", "Skor Kemiripan (Similarity)"]

    recommendations = recommendations.merge(
        book_meta, left_on="Judul Buku", right_index=True, how="left"
    )
    recommendations["Skor Kemiripan (Similarity)"] = recommendations[
        "Skor Kemiripan (Similarity)"
    ].round(4)

    recommendations = recommendations.rename(columns={
        "Book-Author": "Penulis",
        "Year-Of-Publication": "Tahun Terbit",
        "Publisher": "Penerbit"
    })
    return recommendations


# ==========================================
# ANTARMUKA PENGGUNA (STREAMLIT UI)
# ==========================================
st.title("📚 Sistem Rekomendasi Buku")
st.markdown(
    "Aplikasi rekomendasi buku interaktif menggunakan pendekatan **Popularity-Based** "
    "dan **Item-Based Collaborative Filtering** pada dataset *Book-Crossing*."
)

# Sidebar untuk pengaturan sumber dataset
with st.sidebar:
    st.header("⚙️ Pengaturan Dataset")
    use_upload = st.checkbox("Unggah file CSV secara manual", value=False)

    books_file = None
    ratings_file = None

    if use_upload:
        books_file = st.file_uploader("Unggah BX-Books.csv", type=["csv"])
        ratings_file = st.file_uploader("Unggah BX-Book-Ratings.csv", type=["csv"])
    else:
        default_books = "BX-Books.csv"
        default_ratings = "BX-Book-Ratings.csv"
        if os.path.exists(default_books) and os.path.exists(default_ratings):
            books_file = default_books
            ratings_file = default_ratings
            st.success("File lokal `BX-Books.csv` & `BX-Book-Ratings.csv` ditemukan.")
        else:
            st.warning(
                "File `BX-Books.csv` atau `BX-Book-Ratings.csv` tidak ditemukan di folder aplikasi. "
                "Centang opsi di atas untuk mengunggah file secara manual."
            )

# Validasi ketersediaan dataset sebelum menjalankan model
if books_file is None or ratings_file is None:
    st.info("Silakan sediakan file `BX-Books.csv` dan `BX-Book-Ratings.csv` untuk memulai.")
    st.stop()

try:
    df_explicit, df_final, similarity_df, book_meta = load_and_preprocess_data(
        books_file, ratings_file
    )
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca atau memproses dataset: {e}")
    st.stop()

# Ringkasan statistik dataset di sidebar
with st.sidebar:
    st.divider()
    st.subheader("📊 Ringkasan Data")
    st.metric("Baris Rating Terpilih", f"{df_final.shape[0]:,}")
    st.metric("Jumlah Judul Buku (Model)", f"{similarity_df.shape[0]:,}")

# Tab utama aplikasi
tab_populer, tab_mirip = st.tabs([
    "🏆 Top Buku Terpopuler",
    "🔍 Rekomendasi Buku Serupa"
])

# --- TAB 1: POPULARITY-BASED RECOMMENDATION ---
with tab_populer:
    st.subheader("Daftar Buku Terpopuler")
    st.caption("Menampilkan buku dengan rata-rata rating tertinggi berdasarkan jumlah ulasan minimal.")

    col1, col2 = st.columns(2)
    with col1:
        top_n_pop = st.slider("Jumlah buku yang ditampilkan:", min_value=5, max_value=50, value=10, step=5)
    with col2:
        min_ratings_pop = st.slider("Minimal jumlah rating:", min_value=10, max_value=200, value=50, step=10)

    popular_books_df = get_popular_recommendations(
        df_explicit,
        book_meta,
        min_ratings=min_ratings_pop,
        top_n=top_n_pop
    )

    if popular_books_df.empty:
        st.warning("Tidak ada buku yang memenuhi kriteria minimal jumlah rating tersebut.")
    else:
        st.dataframe(
            popular_books_df,
            use_container_width=True,
            hide_index=True
        )

# --- TAB 2: COLLABORATIVE FILTERING (ITEM-BASED) ---
with tab_mirip:
    st.subheader("Cari Buku yang Mirip")
    st.caption("Pilih judul buku untuk mendapatkan rekomendasi buku lain dengan pola penilaian pengguna yang serupa.")

    book_list = list(similarity_df.index)
    default_index = book_list.index("1984") if "1984" in book_list else 0

    col_book, col_n = st.columns([3, 1])
    with col_book:
        selected_book = st.selectbox(
            "Pilih atau ketik judul buku:",
            options=book_list,
            index=default_index
        )
    with col_n:
        top_n_sim = st.number_input(
            "Jumlah rekomendasi:",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

    if st.button("Tampilkan Rekomendasi", type="primary"):
        hasil_rekomendasi = recommend_similar_books(
            selected_book,
            similarity_df,
            book_meta,
            top_n=int(top_n_sim)
        )

        if hasil_rekomendasi is None or hasil_rekomendasi.empty:
            st.warning(f"Buku '{selected_book}' tidak ditemukan dalam matriks rekomendasi.")
        else:
            st.success(f"Menampilkan {len(hasil_rekomendasi)} buku yang paling mirip dengan **{selected_book}**:")
            st.dataframe(
                hasil_rekomendasi,
                use_container_width=True,
                hide_index=True
            )
