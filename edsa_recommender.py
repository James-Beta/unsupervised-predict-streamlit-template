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
st.set_page_config(layout='wide', initial_sidebar_state="collapsed")
# Data handling dependencies
import pandas as pd
import numpy as np
import re
from PIL import Image
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from EDA.eda1 import eda2
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
    page_options = ["Recommender System","Welcome", "Overview", "Explore the Movie Database"]

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
    if page_selection == "Overview":
        st.title("Solution Overview")
        #st.write("Describe your winning approach on this page")
        st.write("The core of our solution; we looked into two approaches for our machine learning processes: Content-Based Filtering and Collaborative Filtering")
        o1, o2 = st.columns(2)
        with o1:
            st.subheader("Collaborative filtering")
            image = Image.open('resources/imgs/collab.jpg')
            st.image(image)
            st.write("Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.")
            st.write("It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user.")
            st.write("It looks at the items they like and combines them to create a ranked list of suggestions.")
        with o2:
            st.subheader("Content-based Filtering")
            image = Image.open('resources/imgs/content.jpg')
            st.image(image)
            st.write("Content-based filtering attempts to guess what a user may like based on that user's activity. The algorithm makes recommendations by using keywords and attributes assigned to objects in a database (e.g., genre and director) and matching them to a user profile.")
            st.write("The user profile is created based on data derived from a user’s actions, such as ratings, movies searched for, and clicks on movie links.")
            st.write("We deplored Singular Value decomposition (SVD: Collaborative model based filtering recommender system) algorithm since it generally perform better on large datasets compared to some other models as it decomposes a matrix into constituents arrays of feature vectors corresponding to each row and each column.")
            st.write("There are a number of algorithms that are promising new frontiers in terms of optimizing recommendation systems")
            st.write("In future we plan to implement deep learning and hybrid recommender system.")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Welcome":
        st2, st3 = st.tabs(["Welcome", "Meet the team"])
        with st2:
            st.title("Hello, welcome to SHUJAA")
            st.subheader('We integrate research into practical, relevant solutions to address business and societal challenges.')
            st.title("The Recommender")
            st.subheader('Digital Revolution has brought about the problem that can be best described as "Too much content, too little time".')
            image = Image.open('resources/imgs/Background.jpg')
            st.image(image, use_column_width = 'True')
            st.subheader('This can make streaming feel like a chore when a user has to peruse through millions of movie titles, synopsis, or trailers to choose what to watch next.')
            st.subheader('Recommender systems solve this issue by helping the user find items of their interest faster.')
            st.subheader('The item provider gets a model that would ensure that movies are delivered to the right user. This is because the recommender system is modeled to;')
            st.subheader('1. identify the most relevant products for each user')
            st.subheader('2. Showcase personalized content to each user')
            st.subheader('3. Suggest top offers and discounts to the right user')
            st.subheader('It is inevitable that our recommender web application will improve user engagement on your platform which is guaranteed to cultivate repeat customers, brand advocacy from loyal customers, a stronger emotional connection, and lead to faster sales')


        with st3:
            st.subheader("Meet the team")
            col1,col2 = st.columns(2)
            with col1:
                st.subheader("Samson Oguntuwase")
                image = Image.open('resources/imgs/Samson.jpg')
                st.image(image )
                link = '[GitHub](http://github.com/sampsonola)'
                st.markdown(link, unsafe_allow_html=True)

            with col2:
                st.subheader('Humphery Ojo')
                image = Image.open('resources/imgs/Humphery.jpg')
                st.image(image)
                link = '[Kaggle](https://www.kaggle.com/princesarzy1st/competitions?tab=completed)'
                st.markdown(link, unsafe_allow_html=True)
            col4,col5, col6 = st.columns(3)
            with col4:
                st.subheader("James Beta")
                image = Image.open('resources/imgs/James.jpg')
                st.image(image)
                link = '[GitHub](http://github.com/James-Beta)'
                st.markdown(link, unsafe_allow_html=True)
            with col5:
                st.subheader("Lista Abutto")
                image = Image.open('resources/imgs/Lista.jpg')
                st.image(image)
                link = '[GitHub](http://github.com/lista)'
                st.markdown(link, unsafe_allow_html=True)
            with col6:
                st.subheader("Joseph Aromeh")
                image = Image.open('resources/imgs/Joseph.jpg')
                st.image(image)
                link = '[GitHub](https://github.com/Romzy01)'
                st.markdown(link, unsafe_allow_html=True)

    if page_selection == "Explore the Movie Database":
        st.subheader("Can't decide on what to watch yet?")
        st.subheader("Let's look into the library and find something entertaining for you.")
        eda2()

if __name__ == '__main__':
    main()
