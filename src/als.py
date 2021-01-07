import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import data_preprocessing
from data_preprocessing import create_utility_matrix

def reading_processing():
    df = pd.read_csv("~/Recommendation-Engine/data/data.csv",encoding='unicode_escape')
    df = df.dropna()
    data = create_utility_matrix(df)
    n_users = data.customer.unique().shape[0]
    n_items = data.sku.unique().shape[0]
    train_data, test_data = train_test_split(data, test_size=0.20)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    return(train_data, test_data, n_items, n_users)

def r_t_matrix(number_users, number_items, data):
    rot = np.zeros((number_users, number_items))
    data_values = data.values
    for line in data_values:
        rot[line[0] - 1, line[1] - 1] = line[2]
    return rot

def index_matrix(data_rot):
    I = data_rot.copy()
    I[I > 0] = 1
    I[I == 1] = 1
    return I

def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))

def training(self_dimensions = 10, self_num_epochs = 2, self_lamda = 0.01, self_train_data,
             self_test_data, self_n_users, self_n_items):

    E_matrix = np.eye(self_dimensions)
    train_errors, test_errors = [], []
    R = r_t_matrix(number_users=n_users, number_items=n_items, data = train_data)
    T = r_t_matrix(number_users=n_users, number_items=n_items, data = test_data)
    I = index_matrix(R)
    I2 =  index_matrix(T)
    m, n = R.shape
    Q = 3 * np.random.rand(self_dimensions, n)
    Q[0, :] = R[R != 0].mean(axis=0)
    P = 3 * np.random.rand(self_dimensions, m)
    # Repeat until convergence
    for epoch in range(self_num_epochs):
        print("Iteration of epoch : {}".format(epoch))
        # Fix Q and estimate P
        for i, Ii in enumerate(I):
            nui = np.count_nonzero(Ii)  # Number of items user i has rated
            if (nui == 0): nui = 1  # Be aware of zero counts!

            # Least squares solution
            Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + self_lamda * nui * E_matrix
            Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
            P[:, i] = np.linalg.solve(Ai, Vi)

        # Fix P and estimate Q
        for j, Ij in enumerate(I.T):
            nmj = np.count_nonzero(Ij)  # Number of users that rated item j
            if (nmj == 0): nmj = 1  # Be aware of zero counts!

            # Least squares solution
            Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + self_lamda * nmj * E_matrix
            Vj = np.dot(P, np.dot(np.diag(Ij), R[:, j]))
            Q[:, j] = np.linalg.solve(Aj, Vj)

        train_rmse = rmse(I, R, Q, P)
        test_rmse = rmse(I2, T, Q, P)
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
    return(P, Q)

def prediction(P, Q, customer_id):

    R_hat = pd.DataFrame(np.dot(P.T,Q))
    R = pd.DataFrame(R)

    ratings = pd.DataFrame(data=R.loc[customer_id,R.loc[customer_id,:] > 0]).head(n=5)
    ratings['Prediction'] = R_hat.loc[customer_id,R.loc[customer_id,:] > 0]
    ratings.columns = ['Actual Rating', 'Predicted Rating']

    predictions = R_hat.loc[16,R.loc[16,:] == 0] # Predictions for movies that the user 17 hasn't rated yet
    top5 = predictions.sort_values(ascending=False).head(n=5)
    recommendations = pd.DataFrame(data=top5)
    recommendations.columns = ['Predicted Rating']
    return(top5)

def main():
    train_data, test_data, n_items, n_users = reading_processing()
    P,Q = training(self_dimensions = 10, self_num_epochs = 2, self_lamda = 0.01,
                   self_train_data = train_data, self_test_data = test_data,
                   self_n_users = n_users, self_n_items = n_items)
    top_preds = prediction(P, Q, customer_id = 16)
    print("CustomerId")
    print(top_preds)