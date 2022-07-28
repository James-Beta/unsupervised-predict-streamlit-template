"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
st.set_page_config(layout='wide')
# Data handling dependencies
import pandas as pd
import numpy as np
import re

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('resources/imgs/Background.png')

# Data Loading
ratings_df = pd.read_csv('resources/data/ratings.csv', index_col='movieId')
movies_df =  pd.read_csv('resources/data/movies.csv', index_col='movieId')
imdb_df =  pd.read_csv('resources/data/imdb_data.csv', index_col='movieId')

title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Welcome", "Explore the Data"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Welcome":
        st.title("Solution Overview")
        st.write("Hello, welcome to S.H.U.J.A.A")


    if page_selection == "Explore the Data":
        def join_df(ratings_df, df1, df2):
            df = df.join(df1, on = 'movieId', how = 'left')
            df = df.join(df2, on = 'movieId', how = 'left')
            df = df.drop(columns = ['timestamp', 'runtime', 'budget'], axis = 1)
            return df
        def get_cast(ratings_df):
            ratings_df = ratings_df.copy()
            ratings_df['title_cast'] = ratings_df['title_cast'].astype(str)
            ratings_df['title_cast'] = ratings_df['title_cast'].map(lambda x: x.split('|'))
            return ratings_df
        def get_genres(ratings_df):
            ratings_df['genres'] = ratings_df['genres'].map(lambda x: x.split('|'))
            return ratings_df
        def genre_list(ratings_df):
            genres = ratings_df['genres'].to_list()
            all = ['All']
            all_genres = list(set([b for c in genres for b in c]))
            return all_genres + all
        def latest_movies(ratings_df):
            ratings_df = ratings_df.copy()
            years = [x for x in ratings_df['release_year']]
            years = years.sort(reverse=True)
            latest_year = years[0]
            ratings_df = ratings_df[ratings_df['release_year'] == latest_year]
            return ratings_df
        def get_release_years(ratings_df):
            ratings_df['release_year'] = ratings_df['title'].map(lambda x: re.findall('\d\d\d\d', x))
            ratings_df['release_year'] = ratings_df['release_year'].apply(lambda x: np.nan if not x else int(x[-1]))
            return ratings_df
        def prep(ratings_df):
            ratings_df = get_cast(ratings_df)
            ratings_df = get_genres(ratings_df)
            ratings_df = get_genres(ratings_df)
            ratings_df = get_release_years(ratings_df)
            return ratings_df
        def count_df(ratings_df, k = 1000):
            ratings_df = ratings_df.dropna()
            ratings_df['frequency'] = ratings_df.groupby('title')['title'].transform('count')
            ratings_df = ratings_df[ratings_df['frequency'] > k]
            ratings_df = ratings_df.drop(columns = ['frequency'], axis = 1)
            return ratings_df
        def get_popular_movies(ratings_df, k = 10):
            popularity = ratings_df.groupby(['title'])['rating'].count()*ratings_df.groupby(['title'])['rating'].mean()
            popularity = popularity.sort_values(ascending=False).head(50)
            pop = popularity[:k].index.to_list()
            return pop
        def get_pop_directors(ratings_df, k = 20):
            ratings_df_dir = ratings_df.groupby(['director'])['rating'].mean().sort_values(ascending = False)
            top_dir = ratings_df_dir[0:k].index.to_list()
            return top_dir
        def year_list(ratings_df):
            genres = ratings_df['release_year'].to_list()
            all = ['All']
            all_genres = list(set([b for c in genres for b in c]))
            return all_genres + all
        ratings_df = join_df(ratings_df, imdb_df, movies_df)
        ratings_df = prep(ratings_df)
        ratings_df = count_df(ratings_df, 1000)
        eda = ["Latest Movies", "Popular Movies", "Popular Directors"]
        eda_selection = st.selectbox("Select feature to explore", eda)
        if eda_selection == "Latest Movies":
            ratings_df_copy = ratings_df.copy()
            ratings1_df = latest_movies(ratings_df_copy)
            st.title("Latest Movies")
            st.write("Explore movies released within the last year")
            gen = genre_list(ratings1_df)
            genre = st.selectbox("Select genre to explore", gen)
            if genre != "All":
                ratings1_df = ratings1_df[[genre in x for x in list(ratings1_df['genres'])]]
            dir = get_pop_directors(ratings1_df, k = 20)
            director = st.selectbox("Select Director:", dir)
            if director != "ALL":
                ratings_df1 = ratings1_df[ratings1_df['director'] == director]
        if eda_selection == "Popular Movies":
            st.title("Popular Movies")
            st.write("")
            ratings_df2 = ratings_df.copy()
            gen = genre_list(ratings_df2)
            genre = st.selectbox("Select genre to explore", gen)
            if genre != "All":
                ratings_df2 = ratings_df2[[genre in x for x in list(ratings_df2['genres'])]]
            dir = get_pop_directors(ratings_df2, k = 20)
            director = st.selectbox("Select Director:", dir)
            if director != "ALL":
                ratings_df2 = ratings_df2[ratings_df2['director'] == director]

            yr = year_list(ratings_df)
            year = st.selectbox("Select release year:", yr)
            if year != 'All':
                ratings_df2 = ratings_df2[ratings_df2['release_year'] == year]
            pop_mov = get_popular_movies(ratings_df2, k = 10)
            st.subheader("The Top 10 most popular movies are:")
            for i,j in enumerate(pop_mov):
                st.write(str(i+1)+'. '+j)
        if eda_selection == "Popular Directors":
            st.title("Popular Directors")
            st.write("Discover more about your favourite directors")
            ratings_df3 = ratings_df.copy()
            gen = genre_list(ratings_df3)
            genre = st.selectbox("Select genre to explore", gen)
            if genre != "All":
                ratings_df3 = ratings_df3[[genre in x for x in list(ratings_df3['genres'])]]
            yr = year_list(ratings_df3)
            year = st.selectbox("Select release year:", yr)
            if year != 'All':
                ratings_df3 = ratings_df3[ratings_df3['release_year'] == year]



if __name__ == '__main__':
    main()
