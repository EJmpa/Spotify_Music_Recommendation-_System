 ## Spotify Music Recommendation System

This Streamlit app allows users to get recommendations for songs, albums, or artists based on their input. 

### Step-by-Step Explanation

#### 1. Import the necessary libraries

```
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
```

#### 2. Set up the Spotify API credentials

```
# Spotify API Credentials
SPOTIPY_CLIENT_ID = ''
SPOTIPY_CLIENT_SECRET = ''

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)
```

#### 3. Create a simple Streamlit app

```
# Create a simple Streamlit app
st.title("Spotify Music Recommendation System")
st.write("Enter a song, album, or artist to get recommendations.")
```

#### 4. Define the function to get recommendations

```
def get_recommendations(query, query_type):
    results = sp.search(q=query, type=query_type)
    if query_type == 'track':
        track_uri = results['tracks']['items'][0]['uri']
        return sp.recommendations(seed_tracks=[track_uri])['tracks']
    elif query_type == 'album':
        album_uri = results['albums']['items'][0]['uri']
        return sp.recommendation_genre_seeds()
    elif query_type == 'artist':
        artist_uri = results['artists']['items'][0]['uri']
        return sp.recommendations(seed_artists=[artist_uri])['tracks']
```

#### 5. Get the user input

```
# Enhanced user input
song_name = st.text_input("Song Name", "Type here...")
album_name = st.text_input("Album Name", "Type here...")
artist_name = st.text_input("Artist Name", "Type here...")
```

#### 6. Display the recommendations

```
if st.button('Search'):
    # Songs
    if song_name:
        st.markdown("### Recommendations based on Song:")
        song_recommendations = get_recommendations(song_name, 'track')

