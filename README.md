
# Spotify Music Recommendation System 

An interactive web application that leverages the Spotify API and machine learning algorithms to provide song recommendations based on mood.

## 🎵 Features

- **Dataset Recommendation**: Recommend songs from a provided dataset based on mood.
- **API Recommendation**: Use the Spotify API to fetch and recommend songs.
- **Data Analysis**: View data visualizations and analyses on various datasets related to Spotify songs.

## 🛠 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/EJmpa/Spotify_Music_Recommendation-_System.git
cd spotify_music_recommendation_system
```

### 2. Set up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate
```

### 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

### 4. Spotify API Setup

- Register your application on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
- Retrieve your `client_id` and `client_secret`.

```bash
export SPOTIPY_CLIENT_ID='your_client_id'
export SPOTIPY_CLIENT_SECRET='your_client_secret'
```

### 5. Run the Streamlit App

```bash
streamlit run app2.py
```

## 📋 Usage

### Dataset Recommendation:
1. Select a song from the dropdown.
2. Choose a mood.
3. Get a list of recommended songs based on your selection.

### API Recommendation:
1. Input a song name.
2. Choose a mood.
3. Get a list of recommended songs fetched from the Spotify API based on your input.

### Data Analysis:
1. Choose a dataset from the sidebar.
2. Load and view various analyses and visualizations based on the dataset.

## 📜 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
