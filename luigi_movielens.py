'''
To run a particular module: 
e.g. "python luigi_movielens.py TrainKMeans"

To run with email notifications: 
e.g. "python luigi_movielens.py --email-force-send TrainNearestNeighbors"

To set multiple workers: 
e.g. "python luigi_movielens.py --workers 2 TrainKMeans"
e.g. "python luigi_movielens.py --workers 4 RunBothModels"

To open Luigi visualization:
In a command line interface, enter "luigid"
In browser, enter http://localhost:8082/static/visualiser/index.html

'''

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

class moviesDataIngestion(luigi.Task):

    def run(self):
        movies = pd.read_csv('ml-latest/movies.csv', header=0)
 
        movies.to_csv('movies.csv', index=False)

    def output(self):
        return luigi.LocalTarget('movies.csv')

class moviesDataIngestionCheck(luigi.Task):

    def requires(self):
        yield moviesDataIngestion()

    def run(self):
        movies = pd.read_csv(moviesDataIngestion().output().path)
 
        #data quality check: NA value, missing values, negative or extremely large values
        data_check(movies)

        movies.to_csv('moviesChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('moviesChecked.csv')

class ratingsDataIngestion(luigi.Task):

    def run(self):
        ratings = pd.read_csv('ml-latest/ratings.csv', header=0)

        ratings.to_csv('ratings.csv', index=False)

    def output(self):
        return luigi.LocalTarget('ratings.csv')

class ratingsDataIngestionCheck(luigi.Task):

    def requires(self):
        yield ratingsDataIngestion()

    def run(self):
        ratings = pd.read_csv(ratingsDataIngestion().output().path)
 
        #data quality check: NA value, missing values, negative or extremely large values
        data_check(ratings)

        ratings.to_csv('ratingsChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('ratingsChecked.csv')

class linksDataIngestion(luigi.Task):

    def run(self):
        links = pd.read_csv('ml-latest/links.csv', header=0)

        links.to_csv('links.csv', index=False)

    def output(self):
        return luigi.LocalTarget('links.csv')

class linksDataIngestionCheck(luigi.Task):

    def requires(self):
        yield linksDataIngestion()

    def run(self):
        links = pd.read_csv(linksDataIngestion().output().path)
 
        #data quality check: NA value, missing values, negative or extremely large values
        data_check(links)

        links.to_csv('linksChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('linksChecked.csv')


class tagsDataIngestion(luigi.Task):

    def run(self):
        tags = pd.read_csv('ml-latest/tags.csv', header=0)

        tags.to_csv('tags.csv', index=False)

    def output(self):
        return luigi.LocalTarget('tags.csv')

class tagsDataIngestionCheck(luigi.Task):

    def requires(self):
        yield tagsDataIngestion()

    def run(self):
        tags = pd.read_csv(tagsDataIngestion().output().path)
 
        #data quality check: NA value, missing values, negative or extremely large values
        data_check(tags)

        tags.to_csv('tagsChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('tagsChecked.csv')

class genome_tagsDataIngestion(luigi.Task):

    def run(self):
        genome_tags = pd.read_csv('ml-latest/genome-tags.csv', header=0)

        genome_tags.to_csv('genome_tags.csv', index=False)

    def output(self):
        return luigi.LocalTarget('genome_tags.csv')

class genome_tagsDataIngestionCheck(luigi.Task):

    def requires(self):
        yield genome_tagsDataIngestion()

    def run(self):
        genome_tags = pd.read_csv(genome_tagsDataIngestion().output().path)
 
        #data quality check: NA value, missing values, negative or extremely large values
        data_check(genome_tags)

        genome_tags.to_csv('genome_tagsChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('genome_tagsChecked.csv')

class genome_scoresDataIngestion(luigi.Task):

    def run(self):
        genome_scores = pd.read_csv('ml-latest/genome-scores.csv', header=0)

        genome_scores.to_csv('genome_scores.csv', index=False)

    def output(self):
        return luigi.LocalTarget('genome_scores.csv')

class genome_scoresDataIngestionCheck(luigi.Task):

    def requires(self):
        yield genome_scoresDataIngestion()

    def run(self):
        genome_scores = pd.read_csv(genome_scoresDataIngestion().output().path)
 
        #data quality check: NA value, missing values, negative or extremely large values
        data_check(genome_scores)

        genome_scores.to_csv('genome_scoresChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('genome_scoresChecked.csv')


class DataPreProcessing1(luigi.Task): 
    
    def requires(self):
        yield moviesDataIngestionCheck()
        yield genome_scoresDataIngestionCheck()
        yield genome_tagsDataIngestionCheck()

    def run(self): 
        movies = pd.read_csv(moviesDataIngestionCheck().output().path)
        genome_scores = pd.read_csv(genome_scoresDataIngestionCheck().output().path)
        genome_tags = pd.read_csv(genome_tagsDataIngestionCheck().output().path)

        data = pd.merge(movies, genome_scores, how='inner', on='movieId')
        data = pd.merge(data, genome_tags, how='inner', on='tagId')
        data = data.sort_values(['movieId','tagId'])
        
        #data = data[:1]
                   
        data.to_csv('preProcessedData1.csv', index=False)

    def output(self): 
        return luigi.LocalTarget('preProcessedData1.csv')

class DataPreProcessing1Check(luigi.Task):

    def requires(self):
        yield DataPreProcessing1()

    def run(self):
        data = pd.read_csv(DataPreProcessing1().output().path)
 
        #data quality check: each movieId should have relevance scores for all the tags
        sample = random.sample(list(data.movieId.unique()),100)
        temp = [list(data.loc[data.movieId==i].tagId) for i in sample]
        compare = [i==[j for j in range(1,len(data.tagId.unique())+1)] for i in temp]
        
        if False in compare: 
            raise Exception("Data quality check failed: Not all movieId's have relevance scores for all the tags")
        
        #check dimension of df
        if len(data) != (len(data.movieId.unique())*len(data.tagId.unique())): 
             raise Exception("Data quality check failed: Incorrect dataframe dimension")

        data.to_csv('preProcessedData1Checked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('preProcessedData1Checked.csv')


class DataPreProcessing2a(luigi.Task): 
    
    def requires(self):
        return DataPreProcessing1Check()

    def run(self): 
        data = pd.read_csv(DataPreProcessing1Check().output().path)
        
        relevance = np.asarray(data.relevance)
        X = np.reshape(relevance,(len(data.movieId.unique()),len(data.tagId.unique())))
        
        #X = X[1:]

        pd.DataFrame(X).to_csv('preProcessedData2a.csv', index=False)

    def output(self): 
        return luigi.LocalTarget('preProcessedData2a.csv')

class DataPreProcessing2aCheck(luigi.Task):

    def requires(self):
        yield DataPreProcessing1Check()
        yield DataPreProcessing2a()

    def run(self):
        data1 = pd.read_csv(DataPreProcessing1Check().output().path)
        data2 = pd.read_csv(DataPreProcessing2a().output().path)
 
        #data quality check: mean score for each movie should equal to the ones calculated pre-transformation
        if max(abs(np.array(list(data1.groupby(data1.movieId).relevance.mean())) - np.array(list(data2.mean(axis=1))))) > 0.001:
            raise Exception("Data quality check failed: Mean score for each movie not equal to the ones pre-transformation")
        
        #check dimension of df
        if data2.shape != (len(data1.movieId.unique()), len(data1.tagId.unique())):
             raise Exception("Data quality check failed: Incorrect dataframe dimension")

        data2.to_csv('DataPreProcessing2aChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('DataPreProcessing2aChecked.csv')


class DataPreProcessing2b(luigi.Task): 
    
    def requires(self):
        yield DataPreProcessing1Check()

    def run(self): 
        data = pd.read_csv(DataPreProcessing1Check().output().path)

        data2 = data.sort_values(by=['tagId','movieId'])[:len(data.movieId.unique())]        
            
        #data2 = data2[1:]
              
        data2.to_csv('preProcessedData2b.csv', index=False)

    def output(self): 
        return luigi.LocalTarget('preProcessedData2b.csv')


class DataPreProcessing2bCheck(luigi.Task):

    def requires(self):
        yield DataPreProcessing1Check()
        yield DataPreProcessing2b()

    def run(self):
        data1 = pd.read_csv(DataPreProcessing1Check().output().path)
        data2 = pd.read_csv(DataPreProcessing2b().output().path)
 
        #check dimension of df
        if len(data2) != len(data1.movieId.unique()):
             raise Exception("Data quality check failed: Incorrect number of rows (movies)")

        data2.to_csv('DataPreProcessing2bChecked.csv', index=False)

    def output(self):
        return luigi.LocalTarget('DataPreProcessing2bChecked.csv')


class TrainKMeans(luigi.Task):

    def requires(self):
        yield moviesDataIngestionCheck()
        yield DataPreProcessing2aCheck()
        yield DataPreProcessing2bCheck()

    def run(self):
        n_clusters = 10 # number of clusters
        n = 20 # number of movies to show in each cluster 
                 
        movies = pd.read_csv(moviesDataIngestionCheck().output().path)
        X = pd.read_csv(DataPreProcessing2aCheck().output().path)
        movie_id_title_map = pd.read_csv(DataPreProcessing2bCheck().output().path)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
     
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
            clusters2[i] = clusters2[i].map(movie_id_title_map.title)
        
        clusters2.columns = ['cluster '+str(i) for i in range(0,n_clusters)]
        
        print(clusters2)
   
        pd.DataFrame(clusters2).to_csv('clusters.csv', index=False)

    def output(self): 
        return luigi.LocalTarget('clusters.csv')

class TrainNearestNeighbors(luigi.Task):

    def requires(self):
        yield DataPreProcessing2aCheck()
        yield DataPreProcessing2bCheck()

    def run(self):
        n = 20 # number of movies to show in each cluster 

        X = pd.read_csv(DataPreProcessing2aCheck().output().path)
        movie_id_title_map = pd.read_csv(DataPreProcessing2bCheck().output().path)

        neighbors = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree').fit(X)
        distances, indices = neighbors.kneighbors(X)
        indices = np.delete(indices,0,1)
        distances = np.delete(distances,0,1)

        indices2 = pd.DataFrame(np.transpose(indices))
        indices2.index += 1
        
        for i in indices2.columns: 
            indices2[i] = indices2[i].map(movie_id_title_map.title)
        
        indices2.columns = [movie_id_title_map.title]
        
        most_similar_movies = indices2
        
        print(most_similar_movies)
        
        pd.DataFrame(most_similar_movies).to_csv('most_similar_movies.csv', index=False)

    def output(self): 
        return luigi.LocalTarget('most_similar_movies.csv')

class RunBothModels(luigi.Task):

    def requires(self):
        yield TrainKMeans()
        yield TrainNearestNeighbors()

    def run(self):
        clusters = pd.read_csv(TrainKMeans().output().path)
        most_similar_movies = pd.read_csv(TrainNearestNeighbors().output().path)
               
        pd.DataFrame(clusters).to_csv('clusters2.csv', index=False)
        pd.DataFrame(most_similar_movies).to_csv('most_similar_movies2.csv', index=False)

    def output(self): 
        yield luigi.LocalTarget('clusters2.csv')
        yield luigi.LocalTarget('most_similar_movies2.csv')


class ForceableTask(luigi.Task): 
    # Delete target files so the module can be re-run
    # Make sure to delete the output csv files too

    force = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # To force execution, we just remove all outputs before `complete()` is called
        if self.force is True:
            outputs = luigi.task.flatten(self.output())
            for out in outputs:
                if out.exists():
                    os.remove(self.output().path)


if __name__ == '__main__':
    luigi.run()

