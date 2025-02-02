import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from contextlib import redirect_stdout
from io import StringIO
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise import SVDpp
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calc_metrics(user_id, preds, unique_movie_ids, df_users):
  
      predictions = preds #[svdpp.predict(user_id, movie_id) for movie_id in unique_movie_ids]
      # Estimation for all predictions
      ests = []
      acts = []
      for pred in predictions:
          rating = df_users[(df_users['userId'] == pred.uid) & (df_users['movieId'] == pred.iid)]['rating'].values

          if len(rating) > 0:
              acts.append(rating[0])
              ests.append(pred.est)

      filtered_ = [prediction for prediction in predictions if prediction.est >= 3.5]
      # recommended_items = set(prediction.iid for prediction in predictions)

      actual_ratings = []
      est_ratings = []
      recommended_indata = []
      preds_rmse = []
      
      for preds in filtered_:
        
        rating = df_users[(df_users['userId'] == preds.uid) & (df_users['movieId'] == preds.iid)]['rating'].values

        if len(rating) > 0:
          recommended_indata.append(preds.iid)
          actual_rating = rating[0]
          actual_ratings.append(actual_rating)
          est_ratings.append(preds.est)
          preds_rmse.append(preds)

      recommended_indata_set = set(recommended_indata)
      relevant_items = set(df_users[df_users['userId'] == user_id][df_users['rating'] >= 3.5]['movieId'])
      relevant_recommendation = recommended_indata_set.intersection(relevant_items) #relevant_items.intersection(recommended_items)
      
      if len(relevant_items) == 0:
        recall = 0
      else:
        recall = len(relevant_recommendation) / len(relevant_items)
      
      if len(recommended_indata) == 0:
        precision = 0
      else:
        precision = len(relevant_recommendation) / len(recommended_indata)
      
      # if len(recommended_indata) == 0:
      #   rmse, mae = -1.0, -1.0
      #   print("No relevant recommendations have been made")
      # else:
      #   rmse = np.sqrt(mean_squared_error(est_ratings, actual_ratings))
      #   mae = mean_absolute_error(est_ratings, actual_ratings)

      if len(ests) > 0:
        rmse = np.sqrt(mean_squared_error(ests, acts))
        mae = mean_absolute_error(ests, acts)
      else:
        rmse = -1.0
        mae = -1.0
      
      return precision, recall, rmse, mae

def get_recommendation(user_id, algo, unique_movie_ids, top_k = 5):
  """
  Get the recommendations
  """
  preds = [algo.predict(user_id, movie_id) for movie_id in unique_movie_ids]
  sorted_predictions = sorted(preds, key=lambda x: x.est, reverse=True)
  top_recommendations = sorted_predictions[:top_k]
  return top_recommendations, preds


if __name__=='__main__':

    
    # Config settings
    isTrain = False
    search = False # Search for the best parameters for the SVDpp

    # Data Overview: Show a brief overview of the initial raw data
    
    df_movies = pd.read_csv("./data/movies.csv")
    df_users = pd.read_csv("./data/ratings.csv")
    
    # Data clean-up
    st.header("Data Overview")
    st.write("In this section, we briefly go over how the data looks.")
    st.subheader("Movie data")
    st.dataframe(df_movies.head())
    st.subheader("User data")
    st.dataframe(df_users.head())
    
    st.subheader("Possible rating values")
    st.write("Ratings can take up ordinal values from 0.5 to 5.0 as evident in the graph.")
    rating_counts = df_users['rating'].value_counts().sort_index()

    # Plot the bar chart
    fig, ax = plt.subplots()
    ax.bar(rating_counts.index, rating_counts.values, color='skyblue')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Ratings')

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.subheader("User-specific rating distribution")
    average_ratings = df_users.groupby('userId')['rating'].mean().sort_values()

    # Plot the bar chart
    fig, ax = plt.subplots()
    ax.bar(average_ratings.index, average_ratings.values, color='skyblue')
    ax.set_xlabel('User ID')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Ratings per User')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Merged data-frame
    st.header("Feature Enhancement")
    st.write(" Both the movie.csv and user.csv files were clean - no duplicates or N/A data. For our purpose, "+
    " data in user.csv is enough to train the recommendation model since we are implementing a collaborative-recommender system "+
    "that depends more on the user's interaction with the item. However, for the readibility aspect, the titles column has been added.")
    
    merged_df = df_users.merge(df_movies, on='movieId')
    merged_df.drop(columns=['timestamp','genres'], axis=1, inplace=True)

    st.dataframe(merged_df.head())

    
    st.header("Data-Set split")

    st.write("As depicted in the following code-snippet, the merged data-set is divided into"+\
    "the training and testing data, by utilizing the sklearn.train_test_split API which divides the "+\
    "dataframe based on the userId column.")

    code = """
train_df, test_df = train_test_split(merged_df, test_size=0.5, 
stratify=merged_df['userId'], random_state=42)
"""

    # Use st.code to display the code
    st.code(code, language='python')

    st.header("Recommendation Abstract")

    # st.write("The recommender used is SVD++. This is a collaborative ")
    # st.write("SVD++ model from the surprise-scikit library has been chosen for the recommendation system. "+
    # "This is because SVD++ is very good at modelling the implicit user behavior throught the movies' already rated by the users in the training set as well as predicting "+
    # " ")

    st.write("SVD++ model from the surprise-scikit library has been chosen for implementing a collaborative-recommendation system. "+
    "This is because SVD++ is very good at modelling the implicit user behavior throught the movies' already rated by the users in the "+
    "training set as well as predicting the ratings for the (user_id, movie_id) pair. This was made apparent by comparing the RMSE and MAE metrics for SVD++, "+
    " SVD and K-Nearest Neighbours available in the surprise library and, SVD++ outperformed the other models. ")

    Accuracy = {
      'Model': ['KNN','SVD', 'SVDpp'],
      'RMSE': [0.9718, 0.8954, 0.8895],
      'MAE': [0.7468, 0.6912, 0.6855]
    }
    
    st.write("Model accuracy on test set")
    st.dataframe(Accuracy)

    unique_movie_ids = df_movies['movieId'].unique().tolist()
    unique_user_ids = df_users['userId'].unique().tolist()

    train_df, test_df = train_test_split(merged_df, test_size=0.5, stratify=merged_df['userId'], random_state=42)

    # Loading training data...
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data, test_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader),\
                            Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader)

    train_data, test_data = train_data.build_full_trainset(), test_data.build_full_trainset()
    train_data=Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)

    # grid search for the best parameter for the SVD++
    if search:
        param_grid = {'n_factors': [20, 50, 100],
                'n_epochs': [10, 20, 30],
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.1, 0.2]}

        algo = SVDpp()
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'], cv=3)
        grid_search.fit(train_data)
        best_params = grid_search.best_params['rmse']
    # train with some params
    elif isTrain and not search:

        # params = {'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}
        params = {'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.01, 'reg_all': 0.1}
        
        algo = SVDpp(**params)

        cross_validate(algo, train_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        with open('svdpp_model.pkl', 'wb') as file:
            pickle.dump(algo, file)
        
       # exit()
    # Evaluation with a saved model
    else:
        algo = SVDpp()
        with open('svdpp_model.pkl', 'rb') as file:
            algo = pickle.load(file)

    # params = {'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}
    # algo = SVDpp(**params)
    # cross_validate(algo, train_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    st.header("Recommendation Demo")
    st.write("Slide to the user_id you want to view!")
    # Recommendation Demo
    selected_user_id = st.slider("Select User ID", min_value=df_users['userId'].min(), max_value=df_users['userId'].max())

    # Button to trigger recommendations
    if st.button("Get Recommendations"):
        
        # Training-data
        st.subheader('Movie ratings for the user from the training-set')

        # Plotting the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(train_df['movieId'], train_df['rating'], color='blue')
        ax.set_xlabel('Movie ID')
        ax.set_ylabel('Rating')
        ax.set_title('Movie ID vs Rating')

        # Display the plot in Streamlit
        st.pyplot(fig)
        
        st.subheader("Top 5 recommendations:")
        # Get top-5 recommendation
        top_recs, preds = get_recommendation(selected_user_id, algo, unique_movie_ids, 5)
        rec_movies =[df_movies[df_movies['movieId'] == pred.iid]['title'] for pred in top_recs]
        st.write(rec_movies)

        st.write(len(preds))

        # Accuracy
        st.subheader("User Recommendation Accuracy:")
        st.write("Both, the training and test data-set has been utilized for calculating these metrics."+
        "Here, root-mean squared error (RMSE) and mean absolute error (MAE) metrics "
        +" are calculated between the actual rating and the estimated rating from the SVD++ model "+ 
        "for the (userId, movieId) pair. These metrics are used to get the error in the rating estimation. "+
        "Precision and recall metrics are used to measure if the recommendations made by model are in accordance to the user's behavior and if the model is able to retain "
        +"the data-points. Precision has been calculated by dividing the relevant recommendations by the total number of recommendations that the model thinks is relevant. "+
        "Similarly, Recall is calculated by dividing the relevant recommendations by the actual relevant (userId, movieId) pair in the whole dataset. Note, here "+
        "relevant (userId, movieId) pair refers to having ratings over 3.5 which was a threshold chosen based on the data-set.")

        precision, recall, rmse, mae = calc_metrics(selected_user_id, preds, unique_movie_ids, df_users)
        st.write("Precision: ", precision)
        st.write("Recall: ", recall)
        if rmse == -1.0:
            st.write("UserID does not any ratings associated with them")
        else:
            st.write("RMSE: ", rmse)
            st.write("MAE: ", mae)

        st.write('Average Precision and Recall Visualization')

        # Plotting the bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['Precision', 'Recall'], [precision, recall], color=['blue', 'green'])
        ax.set_ylim([0.0, 1.0])
        ax.set_title('Average Precision and Recall')
        ax.set_ylabel('Score')

        # Display the plot in Streamlit
        st.pyplot(fig)

