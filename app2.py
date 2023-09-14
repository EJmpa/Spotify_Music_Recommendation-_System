# Library importation
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import warnings

# Suppress the warning from NearestNeighbors
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Spotify API Setup

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="e18fafeb60a949d2a9b7d1efccabe69a",
                                                           client_secret="739bbbed49864382a64a64ccd64ecdcc"))


# Columns to load from the dataset
columns_to_use = ['name', 'artists', 'valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']

# Load dataset
#uploaded_file = st.file_uploader("./data/data.csv", type="csv")
#if uploaded_file is not None:



# Streamlit's cache mechanism to load datasets efficiently
@st.cache_resource
def load_data(filename):
    return pd.read_csv(filename)

data = load_data("./data/data.csv")


# Function to scale features
def scale_features(data):
    features = data[['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']]
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Function to assign moods based on clusters
def assign_mood(cluster):
    if cluster == 0: return "Happy"
    elif cluster == 1: return "Relaxing"
    elif cluster == 2: return "Sad"
    elif cluster == 3: return "Intense"
    else: return "Neutral"

# Taking a subset of the data for faster execution
subset_data = data.sample(n=min(10000, len(data)), random_state=42)

# Scale features for subset
scaled_features_subset = scale_features(subset_data)

# Scale features for the entire dataset and apply KMeans clustering
scaled_features_data = scale_features(data)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features_data)
data['mood'] = data['cluster'].apply(assign_mood)

# Sidebar for page selection
page_selection = st.sidebar.radio("Navigate", ["Dataset Recommendation", "API Recommendation", "Data Analysis"])

if page_selection == "Dataset Recommendation":
    st.title('Mood-based Song Recommendation System from Dataset')

    # Assuming subset_data is initialized before this point
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)

    subset_data['cluster'] = kmeans.fit_predict(scaled_features_subset)
    subset_data['mood'] = subset_data['cluster'].apply(assign_mood)

    # Interactive Widget Setup for the subset
    subset_song_names = subset_data['name'].values
    subset_song_features = subset_data[['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']]
    nn_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute')
    nn_model.fit(subset_song_features)

    # Replace ipywidgets with Streamlit components
    dataset_song_selector = st.selectbox('Select a Song', subset_song_names)
    dataset_mood_selector = st.selectbox('Select Mood', ["Sad", "Neutral", "Happy/Energetic", "Relaxing", "Intense"])

    if st.button('Recommend Songs'):
        # Same logic as the on_button_click function but adapted for Streamlit
        selected_song_features = subset_song_features[subset_data['name'] == dataset_song_selector].values
        distances, indices = nn_model.kneighbors(selected_song_features, n_neighbors=50)
        print(type(indices))
        print(indices)
        collab_recommended = subset_song_names[indices[0]]

        cosine_similarities = cosine_similarity(selected_song_features, subset_song_features)
        content_scores = cosine_similarities[0]
        content_indices = content_scores.argsort()[-50:][::-1]
        content_recommended = subset_song_names[content_indices]

        combined_recommended = np.union1d(collab_recommended, content_recommended)
        selected_mood = dataset_mood_selector

        if selected_mood != "Neutral":
            final_recommendations = subset_data[subset_data['name'].isin(combined_recommended) & (subset_data['mood'] == selected_mood)]['name'].unique()
        else:
            final_recommendations = subset_data[subset_data['name'].isin(combined_recommended)]['name'].unique()

        
        for song in final_recommendations:
            artist_name = data[data['name'] == song]['artists'].values[0]
            st.markdown(f"**Song:** {song} <br> **Artist:** {artist_name}", unsafe_allow_html=True)
            st.write("\n")



elif page_selection == "API Recommendation":
    st.title('Mood-based Song Recommendation System from Spotify API')

      
    # Spotify API Recommendation
    st.header('Spotify API Recommendation')
    spotify_query = st.text_input('Enter a song name:')
    spotify_mood_selector = st.selectbox('Select Spotify Mood', ["Happy", "Relaxing", "Sad", "Intense", "Neutral"])

    # Map moods to genres/keywords
    mood_to_genre = {
        "Happy": "pop OR happy OR dance",
        "Relaxing": "chill OR ambient OR acoustic",
        "Sad": "sad OR melancholy",
        "Intense": "rock OR metal OR intense",
        "Neutral": ""  # No genre filter
    }

    if st.button('Recommend Songs from Spotify'):
        genre_filter = mood_to_genre[spotify_mood_selector]
        if genre_filter:
            search_query = f"{spotify_query} genre:{genre_filter}"
        else:
            search_query = spotify_query  # Neutral mood: no genre filter
        results = sp.search(q=search_query, limit=5)
    
        for idx, track in enumerate(results['tracks']['items']):
            # Display the cover art
            cover_art_url = track['album']['images'][0]['url']
            st.image(cover_art_url, width=200)  # Displaying the cover image with a width of 200 pixels
            
            # Displaying the song name and artist name using markdown for better styling
            st.markdown(f"**{track['name']}** - {track['artists'][0]['name']}")
            
            # Playing the 30s track preview
            st.audio(track['preview_url'])
            
            # Adding spacing after each song for better layout
            st.write("\n\n")

elif page_selection == "Data Analysis":
    st.title('Data Analysis')
    
    dataset_selection = st.sidebar.radio("Choose a dataset for EDA", ["data", "data_w_genres", "data_by_year", "data_by_artist", "data_by_genres"])
    
    if dataset_selection == "data":
        if st.button('Load and Analyze data.csv'):
            data = load_data("./data/data.csv")
            # Dataset Overview and Introduction
            st.title("Spotify Dataset Analysis")
            st.write("""
            **Introduction**

            Spotify, a global leader in music streaming, has revolutionized the way we experience music. With its vast collection spanning genres, moods, and languages, Spotify offers something for every listener. A key factor in Spotify's success is its ability to understand and cater to individual musical preferences. Behind this personalized experience lie sophisticated algorithms and a wealth of data on tracks and user interactions.

            **Dataset Overview**

            The dataset under analysis provides a detailed view of tracks available on Spotify, with the following attributes:

            - **Valence**: A measure indicating the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g., happy, cheerful), while tracks with low valence sound more negative (e.g., sad, depressed).
            - **Year**: The release year of the track.
            - **Acousticness**: A confidence measure indicating whether the track is acoustic.
            - **Artists**: The artists who performed the track.
            - **Danceability**: Describes how suitable a track is for dancing based on a combination of musical elements.
            - **Duration**: The total duration of the track in milliseconds.
            - **Energy**: A measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
            - **Explicit**: Indicates whether the track has explicit content.
            - **Instrumentalness**: Predicts whether a track contains vocals. Higher values mean the track is more likely to be instrumental.
            - **Key**: The key the track is in.
            - **Liveness**: Detects the presence of a live audience in the recording.
            - **Loudness**: The overall loudness of a track in decibels.
            - **Mode**: Modality of the track. Major is represented by 1 and minor is 0.
            - **Name**: The name of the track.
            - **Popularity**: The popularity of the track on Spotify.
            - **Release Date**: The date when the track was released.
            - **Speechiness**: Detects the presence of spoken words in a track.
            - **Tempo**: The overall estimated tempo of a track in beats per minute.

            By analyzing this dataset, we aim to uncover insights into the musical and acoustic properties of tracks, understand trends over the years, and delve into the characteristics that make a track popular.
            """)

            # Data Cleaning    
            def clean_data(data):
                # Removing duplicates based on 'id'
                data_cleaned = data.drop_duplicates(subset='id')
                
                # Convert release_date to datetime format
                data_cleaned['release_date'] = pd.to_datetime(data_cleaned['release_date'], errors='coerce')
                               
                
                # Filtering out rows where artists column is not a list
                data_cleaned = data_cleaned[data_cleaned['artists'].str.startswith('[') & data_cleaned['artists'].str.endswith(']')]
                
                # Converting string representation of list to actual list for 'artists' column
                data_cleaned['artists'] = data_cleaned['artists'].apply(eval)
                
                # Reordering columns
                cols_order = ['id', 'name', 'year', 'artists', 'duration_ms', 'valence', 'acousticness', 'danceability', 'energy', 
                            'instrumentalness', 'liveness', 'loudness', 'popularity', 'speechiness', 'tempo']
                data_cleaned = data_cleaned[cols_order]
                
                return data_cleaned

            st.title("Spotify Dataset Cleaning")
        
            # Displaying the data before cleaning
            st.subheader("Data before cleaning")
            st.write(data)

            # Cleaning the data
            cleaned_data = clean_data(data)

            # Displaying the cleaned data
            st.subheader("Data after cleaning")
            st.write(cleaned_data)

            
            # Descriptive Statistics
            st.subheader('Descriptive Statistics for data.csv')
            st.title('statistical summary of the dataset')
            st.write(cleaned_data.describe())

            
            # Missing Data Visualization using heatmap
            st.subheader('Missing Data Visualization')
            st.write("Yellow areas indicate missing data, while purple areas signify complete data. The heatmap suggests the dataset is relatively complete.")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(cleaned_data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
            st.pyplot(fig)

            # Correlation Matrix Visualization
            st.subheader('Correlation Matrix for Numeric Features')
            st.write("The correlation matrix visualizes the relationships between numeric features. The values in each cell represent the correlation coefficient between two variables.")
            numeric_data = cleaned_data.select_dtypes(include=[np.number])  # Define numeric_data
            corr_matrix = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            st.pyplot(fig)
            #fig, ax = plt.subplots(figsize=(14, 10))
            #corr_matrix = cleaned_data.select_dtypes(include=[np.number]).corr()      
            #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            #st.pyplot(fig)

            # Histograms for Numeric Features
            st.subheader('Histograms for Features')
            st.write("Histograms help in understanding the distribution of data for each numeric feature.")
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                cleaned_data[col].hist(ax=ax, bins=30, edgecolor='black')
                ax.set_title(f'Histogram for {col}')
                st.pyplot(fig)
                        
            
             # Box plots for numerical features
            st.subheader('Box Plots for Numerical Features')
            for column in ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']:
                fig = px.box(data, y=column, title=f'Box plot of {column}')
                st.plotly_chart(fig)
                st.write(f"The box plot for {column} helps identify the central tendency, variability, and possible outliers.")

            # Top artists (if the artists' column is present)
            if 'artists' in data.columns:
                st.subheader('Top Artists')
                top_artists = data['artists'].value_counts().head(10)
                fig = px.bar(top_artists, x=top_artists.index, y=top_artists.values, title='Top 10 Artists', labels={'index': 'Artist', 'value': 'Count'})
                st.plotly_chart(fig)
                st.write("This bar chart displays the top 10 artists based on song counts in the dataset.")




    elif dataset_selection == "data_w_genres":
        if st.button('Load and Analyze data_w_genres.csv'):
            data_w_genres = load_data("./data/data_w_genres.csv")
        
            # Descriptive Statistics
            st.subheader('Descriptive Statistics for data_w_genres.csv')
            st.write(data_w_genres.head())
        
            
        

    elif dataset_selection == "data_by_genres":
        if st.button('Load and Analyze data_by_genres.csv'):
            data_by_genres = load_data("./data/data_by_genres.csv")
            
            # Descriptive Statistics
            st.subheader('Descriptive Statistics for data_by_genres.csv')
            st.write(data_by_genres.head())
            
            
          

    elif dataset_selection == "data_by_artist":
        if st.button('Load and Analyze data_by_artist.csv'):
            data_by_artist = load_data("./data/data_by_artist.csv")
            
            # Descriptive Statistics
            st.subheader('Descriptive Statistics for data_w_artist.csv')
            st.write(data_by_artist.head())
            
            
            
    elif dataset_selection == "data_by_year":
        if st.button('Load and Analyze data_by_year.csv'):
            data_by_year = load_data("./data/data_by_year.csv")
            
            # Descriptive Statistics
            st.subheader('Descriptive Statistics for data_by_year.csv')
            st.write(data_by_year.head())
             



st.write("Thanks for using our application")
