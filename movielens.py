from time import time
import random 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def data_check(df):
    df.info()
    df.describe()
    df.isnull().sum() #count number of null values in each column 
    df.fillna(0) #fix null values (e.g. Replace with 0, mean, etc)

def main():
    #Data ingestion
    movies = pd.read_csv('ml-latest/movies.csv', header=0) 
    ratings = pd.read_csv('ml-latest/ratings.csv', header=0) 
    links = pd.read_csv('ml-latest/links.csv', header=0)
    tags = pd.read_csv('ml-latest/tags.csv', header=0)
    genome_tags = pd.read_csv('ml-latest/genome-tags.csv', header=0)
    genome_scores = pd.read_csv('ml-latest/genome-scores.csv', header=0)
    
    #Data quality check: NA value, missing values, negative or extremely large values
    data_check(movies)
    data_check(ratings)
    data_check(links)
    data_check(tags)
    data_check(genome_tags)
    data_check(genome_scores)
    
    #DataPreProcessing1
    data1 = pd.merge(movies, genome_scores, how='inner', on='movieId')
    data1 = pd.merge(data1, genome_tags, how='inner', on='tagId')
    data1 = data1.sort_values(['movieId','tagId'])
    #data = data[:1]
                       
    #Data quality check: each movieId should have relevance scores for all the tags
    sample = random.sample(list(data1.movieId.unique()),100)
    temp = [list(data1.loc[data1.movieId==i].tagId) for i in sample]
    compare = [i==[j for j in range(1,len(data1.tagId.unique())+1)] for i in temp]
            
    if False in compare: 
        raise Exception("Data quality check failed: Not all movieId's have relevance scores for all the tags")
    
    #check dimension of df
    if len(data1) != (len(data1.movieId.unique())*len(data1.tagId.unique())): 
         raise Exception("Data quality check failed: Incorrect dataframe dimension")
    
    #DataPreProcessing2a
    relevance = np.asarray(data1.relevance)
    data2a = np.reshape(relevance,(len(data1.movieId.unique()),len(data1.tagId.unique())))
    #X = X[1:]
     
    #Data quality check: mean score for each movie should equal to the ones calculated pre-transformation
    if max(abs(np.array(list(data1.groupby(data1.movieId).relevance.mean())) - np.array(list(data2a.mean(axis=1))))) > 0.001:
        raise Exception("Data quality check failed: Mean score for each movie not equal to the ones pre-transformation")
    
    #check dimension of df
    if data2a.shape != (len(data1.movieId.unique()), len(data1.tagId.unique())):
         raise Exception("Data quality check failed: Incorrect dataframe dimension")
    
    #DataPreProcessing2b
    data2b = data1.sort_values(by=['tagId','movieId'])[:len(data1.movieId.unique())]        
                
    #check dimension of df
    if len(data2b) != len(data1.movieId.unique()):
         raise Exception("Data quality check failed: Incorrect number of rows (movies)")
    
    
    #TrainKMeans
    n_clusters = 10 # number of clusters
    n = 20 # number of movies to show in each cluster 
                     
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data2a)
         
    labels = pd.DataFrame(kmeans.labels_)
    labels['movieId'] = labels.index
    labels.reset_index(level=0, inplace=False)
    labels.columns = ['cluster', 'movieId']
    labels2 = pd.merge(labels, movies, how='left', on='movieId')
    labels2.title.fillna(labels2.movieId, inplace=True)
    labels2.genres.fillna(labels2.movieId, inplace=True)
    
    clusters = labels2.groupby('cluster').groups
    clusters2 = dict()
    
    for k in clusters: 
        clusters2[k] = clusters[k][:n]
        
    clusters2 = pd.DataFrame(clusters2)
    clusters2.index += 1
    
    for i in clusters2.columns: 
        clusters2[i] = clusters2[i].map(data2b.title)
    
    clusters2.columns = ['cluster '+str(i) for i in range(0,n_clusters)]
    
    print(clusters2)
       
    pd.DataFrame(clusters2).to_csv('clusters.csv', index=False)
    
    #TrainNearestNeighbors
    n = 20 # number of movies to show in each cluster 
    
    neighbors = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree').fit(data2a)
    distances, indices = neighbors.kneighbors(data2a)
    indices = np.delete(indices,0,1)
    distances = np.delete(distances,0,1)
    
    indices2 = pd.DataFrame(np.transpose(indices))
    indices2.index += 1
    
    for i in indices2.columns: 
        indices2[i] = indices2[i].map(data2b.title)
    
    indices2.columns = [data2b.title]
    
    most_similar_movies = indices2
    
    print(most_similar_movies)
    
    pd.DataFrame(most_similar_movies).to_csv('most_similar_movies.csv', index=False)

if __name__ == '__main__':
    main()

