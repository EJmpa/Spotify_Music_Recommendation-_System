import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API Credentials
SPOTIPY_CLIENT_ID = '5c31e738069944f5bfe87e77c50b4baa'
SPOTIPY_CLIENT_SECRET = 'b7432a163d2b483babef62ec336efd4b'

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Create a simple Streamlit app
st.title("Spotify Music Recommendation System")
st.write("Enter a song, album, or artist to get recommendations.")

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

# Enhanced user input
song_name = st.text_input("Song Name", "Type here...")
album_name = st.text_input("Album Name", "Type here...")
artist_name = st.text_input("Artist Name", "Type here...")

if st.button('Search'):
    # Songs
    if song_name:
        st.markdown("### Recommendations based on Song:")
        song_recommendations = get_recommendations(song_name, 'track')
        for track in song_recommendations:
            col1, col2 = st.columns([1, 3])  # Split the row into two columns: one for image and three for details
            with col1:
                st.image(track['album']['images'][0]['url'], width=50)
            with col2:
                st.markdown(f"**{track['name']}**")
                st.text(track['artists'][0]['name'])
        st.write('---')

    # Albums
    if album_name:
        st.markdown("### Genres based on Album:")
        album_recommendations = get_recommendations(album_name, 'album')
        for genre in album_recommendations['genres']:
            st.text(genre)
        st.write('---')

    # Artists
    if artist_name:
        st.markdown("### Recommendations based on Artist:")
        artist_recommendations = get_recommendations(artist_name, 'artist')
        for track in artist_recommendations:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(track['album']['images'][0]['url'], width=50)
            with col2:
                st.markdown(f"**{track['name']}**")
                st.text(track['artists'][0]['name'])
        st.write('---')
