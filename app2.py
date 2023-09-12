import streamlit as st
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Spotify API Setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="",
                                                           client_secret=""))

# Load dataset
uploaded_file = st.file_uploader("data/data.csv", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)


# Preprocessing and KMeans clustering for mood assignment
features = data[['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

def assign_mood(cluster):
    if cluster == 0: return "Happy"
    elif cluster == 1: return "Relaxing"
    elif cluster == 2: return "Sad"
    elif cluster == 3: return "Intense"
    else: return "Neutral"

data['mood'] = data['cluster'].apply(assign_mood)

# Streamlit UI
st.title('Mood-based Song Recommendation System')

# Dataset Recommendation
st.header('Dataset Recommendation')
dataset_song_selector = st.selectbox('Select a Song', data['name'].values)
dataset_mood_selector = st.selectbox('Select Mood', ["Sad", "Neutral", "Happy", "Relaxing", "Intense"])

if st.button('Recommend Songs from Dataset'):
    recommended_songs = data[data['mood'] == dataset_mood_selector]['name'].sample(5).values
    for song in recommended_songs:
        st.write(song)

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
        st.write(track['name'], '-', track['artists'][0]['name'])
        st.audio(track['preview_url'])  # Play 30s track preview

st.write("Note: Make sure to replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your Spotify app credentials.")
