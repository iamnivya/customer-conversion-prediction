import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.sparse import csr_matrix

#movie recommendation

movies=pd.read_csv("movies.csv")
ratings=pd.read_csv("ratings.csv")
movies.head()
ratings.head()
movies.shape
ratings.shape

#creating pivot table
final_data= ratings.pivot(index='movieId',columns='userId',values='rating')
final_data.fillna(0,inplace=True)
final_data.head()

#to check how many users have voted for particular movie
no_user_voted=ratings.groupby('movieId')['rating'].agg('count')
no_movies_rated=ratings.groupby("userId")['rating'].agg('count')
no_user_voted

#visualizing data of no of user voted for a movie and setting threshold of voting count of a user for that movie
plt.scatter(no_user_voted.index,no_user_voted,color='blue')
plt.axhline(y=10,color='r')
plt.xlabel("MovieId")
plt.ylabel("no of user voted for the movie")

#removing data of user who voted less than 10 movies
final_data=final_data.loc[no_user_voted[no_user_voted>10].index,:]
final_data.shape

#user rated for how many movie
no_movies_rated

#visualizing data of no of movies rated by user and setting threshold as 50
plt.scatter(no_movies_rated.index,no_movies_rated,color='cyan')
plt.axhline(y=50,color='r')
plt.xlabel("userId")
plt.ylabel("No of movies voted")

#removing data of movies rated less than 50 
final_data=final_data.loc[:,no_movies_rated[no_movies_rated>50].index]
final_data.shape

#converting to csr matrix
csr_data=csr_matrix(final_data.values)
final_data.reset_index(inplace=True)

#merging movie id and movie name data in new dataframe
merge_df= pd.merge(final_data['movieId'],movies[['movieId','title']],on='movieId')
merge_df

#building model
knn=NearestNeighbors(metric='eucledian',algorithm='brute',n_neighbors=5,n_jobs=-1)
knn=NearestNeighbors(n_neighbors=5)
knn.fit(csr_data)

#picking the model
with open("knn_model.pkl", "wb") as model_file:
    pickle.dump(knn, model_file)

#def fuction for recommendation
def recommend(movie_name):
    with open("knn_model.pkl", "rb") as model_file:
        knn = pickle.load(model_file)
    no_of_recommandation=5
    idx=process.extractOne(movie_name,movies['title'])[2]
    matrix_data=csr_data[idx]
    print("Movie Selected :",movies['title'][idx])
    print("Searching for Movie  Recommandation based on rating......")
    distance, indices = knn.kneighbors(matrix_data,n_neighbors=no_of_recommandation+1)
    indices=indices[0][1:]
    movie_list=[]
    for i in indices:
        movie_list.append(movies['title'][i])
    df=pd.DataFrame(movie_list,columns=['Recommanded Movie'])
    return df

