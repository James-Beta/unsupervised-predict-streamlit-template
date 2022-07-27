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
movies_df =  pd.read_csv('resources/data/movies.csv', index_col='movieId')
#imdb_df =  pd.read_csv('resources/data/imdb_data.csv', index_col='movieId')

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
        eda = ["Latest Movies", "Popular Movies", "Popular Directors"]
        eda_selection = st.selectbox("Select feature to explore", eda)
#         if eda_selection == "Latest Movies":
#             st.title("Latest Movies")
#             st.write("")
#             col1, col2 = st.columns(2)
#             with col1:
#                 gen = ["All"]
#                 gen1 = ratings_df['genres'].to_list()
#                 gen = gen + gen1
#                 genre = st.selectbox("Select genre to explore", gen)
#                 if genre != "All":
#                     ratings_df = ratings_df[ratings_df['genre'] == genre]

#             with col2:
#                 dir = ['All']
#                 dir1 = ratings_df["director"].to_list()
#                 dir = dir + dir1
#                 director = st.selectbox("Select Director:", dir)
#                 if director != "ALL":
#                     ratings_df = ratings_df[ratings_df['director'] == director]


#         if eda_selection == "Popular Movies":
#             st.title("Popular Movies")
#             st.write("")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 gen = ["All"]
#                 gen1 = ratings_df['genres'].to_list()
#                 gen = gen + gen1
#                 genre = st.selectbox("Select genre to explore", gen)
#                 if genre != "All":
#                     ratings_df = ratings_df[ratings_df['genre'] == genre]

#             with col2:
#                 dir = ['All']
#                 dir1 = ratings_df["director"].to_list()
#                 dir = dir + dir1
#                 director = st.selectbox("Select Director:", dir)
#                 if director != "ALL":
#                     ratings_df = ratings_df[ratings_df['director'] == director]

#             #with col3:

#         if eda_selection == "Popular Directors":
#             st.title("Popular Directors")
#             st.write("")
#             col1, col2 = st.columns(2)
#             with col1:
#                 gen = ["All"]
#                 gen1 = ratings_df['genres'].to_list()
#                 gen = gen + gen1
#                 genre = st.selectbox("Select genre to explore", gen)
#                 if genre != "All":
#                     df = ratings_df[ratings_df['genre'] == genre]

#             #with col2:



if __name__ == '__main__':
    main()
