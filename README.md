# Spotify_Music_Recommendation_System
Spotify Music Recommendation System
An interactive web application that leverages the Spotify API and machine learning algorithms to provide song recommendations based on mood.

Features
Dataset Recommendation: Recommend songs from a provided dataset based on mood.
API Recommendation: Use the Spotify API to fetch and recommend songs.
Data Analysis: View data visualizations and analyses on various datasets related to Spotify songs.
Installation & Setup
Clone the Repository:

bash
Copy code
git clone <repository-link>
cd spotify_music_recommendation-_system
Set up a Virtual Environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the Required Packages:

bash
Copy code
pip install -r requirements.txt
Spotify API Setup:

Register your application on the Spotify Developer Dashboard.
Retrieve your client_id and client_secret.
Set the environment variables:
bash
Copy code
export SPOTIPY_CLIENT_ID='your_client_id'
export SPOTIPY_CLIENT_SECRET='your_client_secret'
Run the Streamlit App:

bash
Copy code
streamlit run app2.py
Usage
Dataset Recommendation:

Select a song from the dropdown.
Choose a mood.
Get a list of recommended songs based on your selection.
API Recommendation:

Input a song name.
Choose a mood.
Get a list of recommended songs fetched from the Spotify API based on your input.
Data Analysis:

Choose a dataset from the sidebar.
Load and view various analyses and visualizations based on the dataset.
Contributing


License
MIT







