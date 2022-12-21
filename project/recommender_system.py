import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

model = joblib.load('model.pkl')

data = pd.read_csv('notebook/data-set/data.csv')
X = data.select_dtypes(np.number)
X.drop(['key'], axis=1, inplace=True)
num_columns = list(X.columns)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# find songs not in the dataset
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track:{} year:{}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None
    
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    
    song_data['name']=[name]
    song_data['year']=[year]
    song_data['explicit']=[int(results['explicit'])]
    song_data['duration_ms']=[results['duration_ms']]
    song_data['popularity']=[results['popularity']]
    
    for key, value in audio_features.items():
        song_data[key] = value
        
    song_data_df = pd.DataFrame(song_data)
    song_cluster_label = model.predict(song_data_df[num_columns])
    song_data_df['cluster'] = song_cluster_label
    
    return song_data_df


def get_song_data(song):
    try:
        song_data = data[(data['name'] == song['name']) & (data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song)
        if song_data is None:
            continue
        song_vector = song_data[num_columns].values
        song_vectors.append(song_vector)
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

def flatten_dict(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    
    return flattened_dict

def recommender(song_list):
    rec_song_list_column = ['name', 'year', 'artists']
    song_dict = flatten_dict(song_list)
    
    song_center = get_mean_vector(song_list)
    scaler = model.steps[0][1]
    scaled_data = scaler.transform(data[num_columns])
    scaled_song_center = scaler.transform(song_center.reshape(1,-1))
    
    distance = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distance)[:, :10][0])
    
    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[rec_song_list_column].to_dict(orient='records')

if __name__ == '__main__':
    print()