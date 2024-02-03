import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

def dataset_cleanup(self):
    # Check for duplicates and if N/A are present
    pass

    

if __name__=='__main__':

    # Data Overview: Show a brief overview of the initial raw data
    
    df_movies = pd.read_csv("data/movies.csv")
    df_users = pd.read_csv("data/ratings.csv")
    
    # Data clean-up
    st.title("Data Overview")
    st.subheader("Movies")
    st.dataframe(df_movies.head())
    # Plot movies avg rating
    # Interest in Genre
    # Plot user-rating 
    # Avg movie ratings
    # Top Genres
    
    # Feature-Enhancement
    # Pivot-Table
    # Merged Table
    merged_df = df_users.merge(df_movies, on='MovieId')
    
    # Train-Test split
    train_df, test_df = train_test_split(merged_df, test_size=0.5, stratify=merged_df['userId'], random_state=42)
