import luigi
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

def moviesDataIngestion():
    movies = pd.read_csv('ml-latest/movies.csv', header=0)
    movies.to_csv('movies.csv', index=False)
    return movies

def moviesDataIngestionCheck(movies):
    data_check(movies)
    movies.to_csv('moviesChecked.csv', index=False)
    return movies

def ratingsDataIngestion():
    ratings = pd.read_csv('ml-latest/ratings.csv', header=0)
    ratings.to_csv('ratings.csv', index=False)
    return ratings

def ratingsDataIngestionCheck(ratings):
    data_check(ratings)
    ratings.to_csv('ratingsChecked.csv', index=False)
    return ratings

def linksDataIngestion():
    links = pd.read_csv('ml-latest/links.csv', header=0)
    links.to_csv('links.csv', index=False)
    return links

def linksDataIngestionCheck(links):
    data_check(links)
    links.to_csv('linksChecked.csv', index=False)
    return links

def tagsDataIngestion():
    tags = pd.read_csv('ml-latest/tags.csv', header=0)
    tags.to_csv('tags.csv', index=False)
    return tags

def tagsDataIngestionCheck(tags):
    data_check(tags)
    tags.to_csv('tagsChecked.csv', index=False)
    return tags

def genome_tagsDataIngestion():
    genome_tags = pd.read_csv('ml-latest/genome-tags.csv', header=0)
    genome_tags.to_csv('genome_tags.csv', index=False)
    return genome_tags

def genome_tagsDataIngestionCheck(genome_tags):
    data_check(genome_tags)
    genome_tags.to_csv('genome_tagsChecked.csv', index=False)
    return genome_tags

def genome_scoresDataIngestion():
    genome_scores = pd.read_csv('ml-latest/genome-scores.csv', header=0)
    genome_scores.to_csv('genome_scores.csv', index=False)
    return genome_scores

def genome_scoresDataIngestionCheck(genome_scores):
    data_check(genome_scores)
    genome_scores.to_csv('genome_scoresChecked.csv', index=False)
    return genome_scores

def DataPreProcessing1(movies, genome_score, genome_tags): 

    data = pd.merge(movies, genome_scores, how='inner', on='movieId')
    data = pd.merge(data, genome_tags, how='inner', on='tagId')
    data = data.sort_values(['movieId','tagId'])
    
    #data = data[:1]
               
    data.to_csv('preProcessedData1.csv', index=False)
    
    return data

def DataPreProcessing1Check(data1):
    
    #data quality check: each movieId should have relevance scores for all the tags
    sample = random.sample(list(data1.movieId.unique()),100)
    temp = [list(data1.loc[data1.movieId==i].tagId) for i in sample]
    compare = [i==[j for j in range(1,len(data1.tagId.unique())+1)] for i in temp]
    
    if False in compare: 
        raise Exception("Data quality check failed: Not all movieId's have relevance scores for all the tags")
    
    #check dimension of df
    if len(data1) != (len(data1.movieId.unique())*len(data1.tagId.unique())): 
         raise Exception("Data quality check failed: Incorrect dataframe dimension")

    data1.to_csv('preProcessedData1Checked.csv', index=False)
    
    return data1 

def DataPreProcessing2a(data1checked): 
    
    relevance = np.asarray(data1checked.relevance)
    data1a = np.reshape(relevance,(len(data1checked.movieId.unique()),len(data1checked.tagId.unique())))
    
    #X = X[1:]

    pd.DataFrame(data1a).to_csv('preProcessedData2a.csv', index=False)

    return data1a 

def DataPreProcessing2aCheck(data1Checked, data2a):

    #data quality check: mean score for each movie should equal to the ones calculated pre-transformation
    if max(abs(np.array(list(data1Checked.groupby(data1Checked.movieId).relevance.mean())) - np.array(list(data2a.mean(axis=1))))) > 0.001:
        raise Exception("Data quality check failed: Mean score for each movie not equal to the ones pre-transformation")
    
    #check dimension of df
    if data2a.shape != (len(data1Checked.movieId.unique()), len(data1Checked.tagId.unique())):
         raise Exception("Data quality check failed: Incorrect dataframe dimension")

    pd.DataFrame(data2a).to_csv('DataPreProcessing2aChecked.csv', index=False)
    
    return data2a

def DataPreProcessing2b(data1Checked): 
    
    data2b = data1Checked.sort_values(by=['tagId','movieId'])[:len(data1Checked.movieId.unique())]        
        
    #data2 = data2[1:]
          
    data2b.to_csv('preProcessedData2b.csv', index=False)
    
    return data2b

def DataPreProcessing2bCheck(data1Checked, data2b):

    #check dimension of df
    if len(data2b) != len(data1Checked.movieId.unique()):
         raise Exception("Data quality check failed: Incorrect number of rows (movies)")

    data2b.to_csv('DataPreProcessing2bChecked.csv', index=False)
    
    return data2b 

def TrainKMeans(movies, data2aChecked, data2bChecked):

    n_clusters = 10 # number of clusters
    n = 20 # number of movies to show in each cluster 
             
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data2aChecked)
 
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
        clusters2[i] = clusters2[i].map(data2bChecked.title)
    
    clusters2.columns = ['cluster '+str(i) for i in range(0,n_clusters)]
    
    print(clusters2)
   
    pd.DataFrame(clusters2).to_csv('clusters.csv', index=False)

def TrainNearestNeighbors(data2aChecked, data2bChecked):

    n = 20 # number of movies to show in each cluster 

    neighbors = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree').fit(data2aChecked)
    distances, indices = neighbors.kneighbors(data2aChecked)
    indices = np.delete(indices,0,1)
    distances = np.delete(distances,0,1)

    indices2 = pd.DataFrame(np.transpose(indices))
    indices2.index += 1
    
    for i in indices2.columns: 
        indices2[i] = indices2[i].map(data2bChecked.title)
    
    indices2.columns = [data2bChecked.title]
    
    most_similar_movies = indices2
    
    print(most_similar_movies)
    
    pd.DataFrame(most_similar_movies).to_csv('most_similar_movies.csv', index=False)

if __name__ == '__main__':
    
    #Data ingestion and check 
    movies = moviesDataIngestion()
    moviesChecked = moviesDataIngestionCheck(movies) 
    ratings = ratingsDataIngestion()
    ratingsChecked = ratingsDataIngestionCheck(ratings)
    links = linksDataIngestion()
    linksChecked = linksDataIngestionCheck(links)
    tags = tagsDataIngestion()
    tagsChecked = tagsDataIngestionCheck(tags)
    genome_tags = genome_tagsDataIngestion()
    genome_tagsChecked = genome_tagsDataIngestionCheck(genome_tags)
    genome_scores = genome_scoresDataIngestion()
    genome_scoresChecked = genome_scoresDataIngestionCheck(genome_scores)
    
    #Data processing 
    data1 = DataPreProcessing1(moviesChecked, genome_scoresChecked, genome_tagsChecked)
    data1Checked = DataPreProcessing1Check(data1)
    data2a = DataPreProcessing2a(data1Checked)
    data2b = DataPreProcessing2b(data1Checked) 
    data2aChecked = DataPreProcessing2aCheck(data1Checked, data2a)
    data2bChecked = DataPreProcessing2bCheck(data1Checked, data2b)
    
    #Modeling 
    TrainKMeans(moviesChecked, data2aChecked, data2bChecked)
    
    TrainNearestNeighbors(data2aChecked, data2bChecked)
    
    
    
