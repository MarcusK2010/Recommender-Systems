import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, KNNBasic, accuracy, SVD, SVDpp
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split, cross_validate


def gd_path(file_id):
    """Generate a shareable link from Google Drive file id."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def pop_based_rec(n, war, wnr):
    weight_avg_rating = war
    weight_num_ratings = wnr

    agg_ratings_df = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    agg_ratings_df.rename(columns={'mean': 'avg', 'count': 'count'}, inplace=True)

    movies_recommendation_df = pd.merge(movies_df, agg_ratings_df, on='movieId', how='left')

    max_possible_avg_rating = 5.0  
    max_possible_num_ratings = agg_ratings_df['count'].max() 

    movies_recommendation_df['weighted_avg'] = (movies_recommendation_df['avg'] * weight_avg_rating) / max_possible_avg_rating
    movies_recommendation_df['weighted_count'] = (movies_recommendation_df['count'] * weight_num_ratings) / max_possible_num_ratings
    movies_recommendation_df['combined_score'] = movies_recommendation_df['weighted_avg'] + movies_recommendation_df['weighted_count']

    top_recommendations = movies_recommendation_df.sort_values(by='combined_score', ascending=False).head(n)[["movieId", "title", "genres"]]
    return top_recommendations


def item_based_rec(title, n):

    user_movie_matrix = pd.pivot_table(data=ratings_df,
                                   values="rating",
                                   index="userId",
                                   columns="movieId",
                                   fill_value=0)

    movies_cosine_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                      columns=user_movie_matrix.columns,
                                      index=user_movie_matrix.columns)

    movie_ratings_df = pd.merge(movies_df, ratings_df, on='movieId', how='left')
    movie_title_mask = movie_ratings_df["title"].str.contains(title, case=False)
    movie_id = movie_ratings_df.loc[movie_title_mask, "movieId"].values[0]
    movie_cosine_df = pd.DataFrame(movies_cosine_matrix[movie_id])
    movie_cosine_df = movie_cosine_df.rename(columns={movie_id: "movie_cosine"})
    movie_cosine_df = movie_cosine_df[movie_cosine_df.index != movie_id]
    movie_cosine_df = movie_cosine_df.sort_values(by="movie_cosine", ascending=False)
    no_of_users_rated_both_movies = [sum((user_movie_matrix[movie_id] > 0) & (user_movie_matrix[movieId] > 0)) for movieId in movie_cosine_df.index]
    movie_cosine_df["users_who_rated_both_movies"] = no_of_users_rated_both_movies
    movie_cosine_df = movie_cosine_df[movie_cosine_df["users_who_rated_both_movies"] > 1]

    movie_info_columns = ["movieId", "title", "genres"]

    movie_cosine_top_n = (movie_cosine_df
                            .head(n)
                            .reset_index()
                            .merge(movie_ratings_df.drop_duplicates(subset="movieId"),
                                    on="movieId",
                                    how="left")
                            [movie_info_columns])

    return movie_cosine_top_n.head(n)


def user_based_rec(user_id):
    movie_ratings_df = ratings_df #pd.merge(movies_df, ratings_df, on='movieId', how='left')
    data = movie_ratings_df[["userId", "movieId", "rating"]]
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data, reader)
    
    trainset, testset = train_test_split(data, test_size=0.2, random_state=142)
    
    full_train = data.build_full_trainset()
    algo = SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)
    algo.fit(trainset)
    
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    
    def get_top_n(predictions, user_id, n=10):

      user_recommendations = []

      for uid, iid, true_r, est, _ in predictions:
          if user_id == uid:
              user_recommendations.append((iid, est))
          else:
              continue

      ordered_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)

      ordered_recommendations_top_n = ordered_recommendations[:n]

      return ordered_recommendations_top_n
      
    top_n = get_top_n(predictions, user_id, n)
    
    tuples_df = pd.DataFrame(top_n, columns=["movieId", "estimated_rating"])

    reduced_df = movies_df.drop_duplicates(subset='movieId').copy()

    tuples_df_expanded = tuples_df.merge(reduced_df, on="movieId", how='left')

    tuples_df_expanded = tuples_df_expanded[['movieId', 'title', "genres"]]

    return tuples_df_expanded


st.checkbox("Use container width", value=True, key="use_container_width")

file_ids = {
    "links": "1PUF22QKakFa62dziMvNR5Yr889m_WOQk",
    "movies": "1dabe2aaN3qgOoxgeYN0uFHc8ePnT7jL4",
    "ratings": "1umn-ZA_bvKg_IVQ3BCrxXo4hIVSfwId3",
    "tags": "1QeDe15jKOGgQmcGPBK1M2lB-nKCHF3B8",
}

links_df = pd.read_csv(gd_path(file_ids["links"]), sep=",")
movies_df = pd.read_csv(gd_path(file_ids["movies"]), sep=",")
ratings_df = pd.read_csv(gd_path(file_ids["ratings"]), sep=",")
tags_df = pd.read_csv(gd_path(file_ids["tags"]), sep=",")

movies_df.drop_duplicates(subset=["title"], inplace=True)

n = 5
title = ""
war = 0
wnr = 0

st.title(" Welcome to WBSFLIX - Movie Recommender")
with st.form(key="my_form"):
    user_id = int(st.number_input("Please put in any User ID: "))
    weight_choice = st.text_input("What do you favour: (H)igh ratings, (P)opular movies, or the (G)olden mean: ").lower()
    if weight_choice == "h":
        war = 0.9
        wnr = 0.1
    elif weight_choice == "p":
        war = 0.1
        wnr = 0.9
    elif weight_choice == "g":
        war = 0.5
        wnr = 0.5
    else:
        st.write("Please enter a valid choice: 'H', 'P', or 'G'.")
    title = st.text_input("Enter any movie title: ")
    submitted = st.form_submit_button("Continue...")

if submitted:
    st.write(f"Popularity-based recommender based on user preference '{weight_choice}'")
    top_recommendations = pop_based_rec(n, war, wnr)
    st.dataframe(top_recommendations, use_container_width=st.session_state.use_container_width, hide_index=True)

    if title:    
        st.write(f"Unit-based recommender based on the movie '{title}'")
        item_based_rec_movies = item_based_rec(title, n)
        st.dataframe(item_based_rec_movies, use_container_width=st.session_state.use_container_width, hide_index=True)

    st.write(f"User-based recommender based on User ID '{user_id}'")
    user_based_rec_movies = user_based_rec(user_id)
    st.dataframe(user_based_rec_movies, use_container_width=st.session_state.use_container_width, hide_index=True)