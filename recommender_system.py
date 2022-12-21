import pandas as pd
import os
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from yellowbrick.target import FeatureCorrelation
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.manifold import TSNE
import pickle
from importants import secretCode,getId

model = pickle.load(open('model.pkl' , 'rb'))
genre_data = pd.read_csv("data-set/data_by_genres.csv")
data = pd.read_csv("data-set/data.csv")
x = genre_data.select_dtypes(np.number)

song_cluster_pipeline = Pipeline([('scaler',StandardScaler()),('kmeans',KMeans(n_clusters = 100 , verbose = False))])
number_cols = list(x.columns)
song_cluster_pipeline.fit(data[number_cols])
song_cluster_labels = song_cluster_pipeline.predict(data[number_cols])
data['cluster_labels'] = song_cluster_labels
data.head(2)

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
sp = spotipy.Spotify(auth_manager = SpotifyClientCredentials(client_id=getId , client_secret=secretCode))
def findSong(name,year):
    songData = defaultdict()
    results = sp.search(q='track:{} year:{}'.format(name,year),limit=1)
    if results['tracks']['items'] == [] :
        return None
    results = results['tracks']['items'][0]
    trackId = results['id']
    audio_features = sp.audio_features(trackId)[0]
    
    songData['name'] = [name]
    songData['year'] = [year]
    songData['explicit'] = [int(results['explicit'])]
    songData['duration_ms'] = [results['duration_ms']]
    songData['popularity'] = [results['popularity']]
    
    for key,value in audio_features.items():
        songData[key] = value
    
    song_data_df = pd.DataFrame(songData)
    song_cluster_label = song_cluster_pipeline.predict(song_data_df[number_cols])
    song_data_df['cluster_label'] = song_cluster_label
    return song_data_df

    from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

def get_song_data(song):
    try:
        song_data = data[(data['name'] == song['name']) & (data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return findSong(song['name'] , song['year'])
    
def get_mean_vector(song_list) :
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song)
        if song_data is None:
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix , axis = 0)
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key,value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommender(song_list):
    rec_song_list_column = ['name' , 'year' , 'artists']
    song_dict = flatten_dict_list(song_list)
    song_centre = get_mean_vector(song_list)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(data[number_cols])
    scaled_song_center = scaler.transform(song_centre.reshape(1,-1))
    distance = cdist(scaled_song_center,scaled_data , 'cosine')
    index = list(np.argsort(distance)[:,:10][0]) #uses quick sort algorithm
    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[rec_song_list_column].to_dict(orient = 'records')