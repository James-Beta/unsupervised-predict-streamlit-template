import streamlit as st
import pandas as pd
import numpy as np
import re
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
st.set_option('deprecation.showPyplotGlobalUse', False)

import base64
def eda2():
    stopwords = set(STOPWORDS)
    #let us use the train.csv to get more ratings and improve the experience for the application
    ratings_df = pd.read_csv('resources/data/ratings.csv', index_col='movieId')
    movies_df =  pd.read_csv('resources/data/movies.csv', index_col='movieId')
    imdb_df =  pd.read_csv('resources/data/imdb_data.csv', index_col='movieId')
    def get_cast(ratings_df):
        ratings_df = ratings_df.copy()
        ratings_df['title_cast'] = ratings_df['title_cast'].map(lambda x: x.split('|'))
        return ratings_df
    def get_genres(ratings_df):
        ratings_df = ratings_df.copy()
        ratings_df['genres'] = ratings_df['genres'].map(lambda x: x.split('|'))
        return ratings_df

    def latest_movies(ratings_df):
        ratings_df = ratings_df.copy()
        years = list(set([x for x in ratings_df['release_year']]))
        years.sort(reverse=True)
        latest_year = years[0:3]
        ratings_df = ratings_df[ratings_df['release_year'].isin(latest_year)]
        return ratings_df
    def get_release_years(ratings_df):
        ratings_df = ratings_df.copy()
        ratings_df['release_year'] = ratings_df['title'].map(lambda x: re.findall('\d\d\d\d', x))
        ratings_df['release_year'] = ratings_df['release_year'].apply(lambda x: np.nan if not x else int(x[-1]))
        return ratings_df
    def prep(ratings_df):
        ratings_df = ratings_df.dropna()
        ratings_df['title_cast'] = ratings_df['title_cast'].astype(str)
        ratings_df['genres'] = ratings_df['genres'].astype(str)
        ratings_df['director'] = ratings_df['director'].astype(str)
        ratings_df = get_cast(ratings_df)
        ratings_df = get_genres(ratings_df)
        ratings_df = get_release_years(ratings_df)
        return ratings_df
    def count_df(ratings_df, k = 100):
        ratings_df = ratings_df.copy()
        ratings_df['frequency'] = ratings_df.groupby('title')['title'].transform('count')
        ratings_df = ratings_df[ratings_df['frequency'] > k]
        ratings_df = ratings_df.drop(columns = ['frequency'], axis = 1)
        return ratings_df
    def get_popular_movies(ratings_df, k = 10):
        popularity = ratings_df.groupby(['title'])['rating'].count()*ratings_df.groupby(['title'])['rating'].mean()
        popularity = popularity.sort_values(ascending=False).head(k)
        pop = popularity[:k].index.to_list()
        return pop
    def get_popular_genres(ratings_df, k = 10):
        popularity = ratings_df.groupby(['genres'])['rating'].count()*ratings_df.groupby(['genres'])['rating'].mean()
        popularity = popularity.sort_values(ascending=False).head(k)
        #pop = popularity[:k].index.to_list()
        return popularity
    def get_pop_directors(ratings_df, k = 20):
        ratings_df_dir = ratings_df.groupby(['director'])['rating'].mean().sort_values(ascending = False)
        top_dir = ratings_df_dir[0:k].index.to_list()
        all = ['All']
        return all + top_dir
    def genre_list(ratings_df):
        ratings_df = ratings_df.copy()
        genres = ratings_df['genres'].to_list()
        all = ['All']
        all_genres = list(set([b for c in genres for b in c]))
        return all + all_genres
    def year_list(ratings_df):
        genres = ratings_df['release_year'].to_list()
        all = ['All']
        all_genres = list(set(genres))
        return all + all_genres
    def director_list(ratings_df):
        ratings_df = ratings_df.copy()
        genres = ratings_df['director'].to_list()
        all = ['All']
        all_genres = list(set(genres))
        return all + all_genres
    ratings_df = ratings_df.join(imdb_df, on = 'movieId', how = 'left')
    ratings_df = ratings_df.join(movies_df, on = 'movieId', how = 'left')
    ratings_df = ratings_df.drop(columns = ['timestamp', 'budget'], axis = 1)
    ratings_df = prep(ratings_df)
    ratings_df = count_df(ratings_df)
    tab1,tab2,tab3 = st.tabs(["Latest Movies", "Popular Movies", "Popular Directors"])
    #eda_selection = st.selectbox("Select feature to explore", eda)
    with tab1:
        ratings_df_copy = ratings_df.copy()
        ratings_df1 = latest_movies(ratings_df_copy)
        st.title("Latest Movies")
        st.write("Explore movies released within the last year")
        gen1 = genre_list(ratings_df1)
        genre1 = st.selectbox("Select genre to explore", gen1, key = "g1")
        if genre1 != "All":
            ratings_df1 = ratings_df1[[genre1 in x for x in list(ratings_df1['genres'])]]
        else:
            ratings_df1 = ratings_df1
        dir1 = director_list(ratings_df1)
        director1 = st.selectbox("Select Director:", dir1, key = "d1")
        if director1 != "All":
            ratings_df1 = ratings_df1[ratings_df1['director'] == director1]
        else:
            ratings_df1 = ratings_df1
        st.subheader("Lets also look at the most active actors")
        cast = ratings_df1['title_cast'].to_list()
        cast_words = [b for c in cast for b in c]
        cast_words = ' '.join(cast_words)
        wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(cast_words)
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        st.pyplot(plt.show())
        st.subheader("Finally lets look at the distribution of the movies across the years")
        active_years = ratings_df1['release_year'].to_list()
        dir_years = [str(b) for b in active_years]
        if len(list(set(dir_years)))>0:
            count = pd.Series(dir_years).value_counts()
            years_df = pd.DataFrame(count)
            st.bar_chart(years_df)
        else:
            st.subheader(f"{len(dir_years)} movies in {dir_years[0]}")

    with tab2:
        st.title("Popular Movies")
        st.write("")
        ratings_df2 = ratings_df.copy()
        gen2 = genre_list(ratings_df2)
        genre2 = st.selectbox("Select genre to explore", gen2, key = "g2")
        if genre2 != "All":
            ratings_df2 = ratings_df2[[genre2 in x for x in list(ratings_df2['genres'])]]
        else:
            ratings_df2 = ratings_df.copy()
        yr2 = year_list(ratings_df2)
        year2 = st.selectbox("Select release year:", yr2, key = "y2")
        if year2 != 'All':
            ratings_df2 = ratings_df2[ratings_df2['release_year'] == year2]
        else:
            ratings_df2 = ratings_df2
        if ((year2 == 'All') and (genre2 == 'All')):
            pop_mov2 = get_popular_movies(ratings_df, k = 10)
            st.subheader("The top 10 most popular movies of all time are:")
            for i,j in enumerate(pop_mov2):
                st.write(str(i+1)+'. '+j)
            st.write("We can see that the you are very familiar with the most popular movies")
            st.write("Although this is no surprise, did you know that in the most popular film, The Shawshank Redemption, Andy and Red’s opening chat in the prison yard – in which Red is pitching a baseball – took 9 hours to shoot. Morgan Freeman pitched that baseball for the entire 9 hours without a word of complaint.")
            st.subheader("Lets also look at the most popular Actors")
            cast = ratings_df['title_cast'].to_list()
            cast_words = [b for c in cast for b in c]
            cast_words = ' '.join(cast_words)
            wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(cast_words)
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
            st.pyplot(plt.show())
            st.subheader("Finally lets look at the distribution of the movies across the years")
            active_years = ratings_df2['release_year'].to_list()
            dir_years = [str(b) for b in active_years]
            if len(list(set(dir_years)))>0:
                count = pd.Series(dir_years).value_counts()
                years_df = pd.DataFrame(count)
                st.line_chart(years_df)
            else:
                st.subheader(f"{len(dir_years)} movies in {dir_years[0]}")
        else:
            pop_mov2 = get_popular_movies(ratings_df2, k = 10)
            st.subheader("Based on your selection the most popular movies are:")
            for i,j in enumerate(pop_mov2):
                st.write(str(i+1)+'. '+j)
            st.subheader("Lets also look at the most popular Actors")
            cast = ratings_df2['title_cast'].to_list()
            cast_words = [b for c in cast for b in c]
            cast_words = ' '.join(cast_words)
            wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(cast_words)
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
            st.pyplot(plt.show())
            st.subheader("Finally lets look at the distribution of the movies across the years")
            active_years = ratings_df2['release_year'].to_list()
            dir_years = [str(b) for b in active_years]
            if len(list(set(dir_years)))>0:
                count = pd.Series(dir_years).value_counts()
                years_df = pd.DataFrame(count)
                st.line_chart(years_df)
            else:
                st.subheader(f"{len(dir_years)} movies in {dir_years[0]}")
    with tab3:
        st.title("Popular Directors")
        st.write("Discover more about your favourite directors")
        ratings_df3 = ratings_df.copy()
        average_runtime = ratings_df3['runtime'].mean()
        gen3 = genre_list(ratings_df3)
        genre3 = st.selectbox("Select genre to explore", gen3, key = "g3")
        if genre3 != "All":
            ratings_df3 = ratings_df3[[genre3 in x for x in list(ratings_df3['genres'])]]
        else:
            ratings_df3 = ratings_df.copy()
        yr3 = year_list(ratings_df3)
        year3 = st.selectbox("Select release year:", yr3, key = "y3")
        if year3 != 'All':
            ratings_df3 = ratings_df3[ratings_df3['release_year'] == year3]
        else:
            ratings_df3 = ratings_df3
        ratings_df_dir = ratings_df3.groupby(['director'])['rating'].mean().sort_values(ascending = False)
        popular = ratings_df_dir[:20].index.to_list()
        st.subheader("The most popular dirctors based on your selection are listed below;")
        st.subheader("Lets focus on your director of choice")
        director = sys = st.radio("Select Director:", popular)
        ratings_df3 = ratings_df3[ratings_df3['director'] == director]
        runtime = (ratings_df3['runtime'].sum())/60
        st.subheader(f"You have {runtime} hours of content from {director}")
        st.subheader(f"{director} has been most active in the following years")
        active_years = ratings_df3['release_year'].to_list()
        dir_years = [str(b) for b in active_years]
        if len(list(set(dir_years)))>0:
            count = pd.Series(dir_years).value_counts()
            years_df = pd.DataFrame(count)
            st.line_chart(years_df)
        else:
            st.subheader(dir_years[0])
        cast = ratings_df1['genres'].to_list()
        cast_words = [b for c in cast for b in c]
        count = pd.Series(cast_words).value_counts()
        genres_df = pd.DataFrame(count)
        st.subheader('Lets investigate their favourite genres')
        st.bar_chart(genres_df)
