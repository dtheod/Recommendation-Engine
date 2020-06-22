import pandas as pd
import numpy as np
import pickle
import os

PIKp = "P_pickle.dat"
PIKq = "Q_pickle.dat"

def r_t_matrix(number_users, number_items, data):
    rot = np.zeros((number_users, number_items))
    for line in data.itertuples():
        rot[line[1] - 1, line[2] - 1] = line[3]
    return rot

def index_matrix(data_rot):
    I = data_rot.copy()
    I[I > 0] = 1
    I[I == 1] = 1
    return I

def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))



class ALS_training(object):

    def __init__(self,
                 lamda,
                 num_epochs,
                 dimensions,
                 train_data,
                 test_data,
                 n_users,
                 n_items):
        self.lamda = lamda
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.test_data = test_data
        self.dimensions = dimensions
        self.n_users = n_users
        self.n_items = n_items
        self.als_training()

    def als_training(self):
        """
        Initialisation of variables
        """
        E_matrix = np.eye(self.dimensions)
        train_errors, test_errors = [], []
        R = r_t_matrix(number_users=self.n_users, number_items=self.n_items, data = self.train_data)
        T = r_t_matrix(number_users=self.n_users, number_items=self.n_items, data = self.test_data)
        I = index_matrix(R)
        I2 =  index_matrix(T)
        m, n = R.shape
        Q = 3 * np.random.rand(self.dimensions, n)
        Q[0, :] = R[R != 0].mean(axis=0)
        P = 3 * np.random.rand(self.dimensions, m)

        # Repeat until convergence
        for epoch in range(self.num_epochs):
            print("Iteration of epoch : {}".format(epoch))
            # Fix Q and estimate P
            for i, Ii in enumerate(I):
                print(i)
                nui = np.count_nonzero(Ii)  # Number of items user i has rated
                if (nui == 0): nui = 1  # Be aware of zero counts!

                # Least squares solution
                Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + self.lamda * nui * E_matrix
                Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
                P[:, i] = np.linalg.solve(Ai, Vi)

            # Fix P and estimate Q
            for j, Ij in enumerate(I.T):
                print(j)
                nmj = np.count_nonzero(Ij)  # Number of users that rated item j
                if (nmj == 0): nmj = 1  # Be aware of zero counts!

                # Least squares solution
                Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + self.lamda * nmj * E_matrix
                Vj = np.dot(P, np.dot(np.diag(Ij), R[:, j]))
                Q[:, j] = np.linalg.solve(Aj, Vj)

            train_rmse = rmse(I, R, Q, P)
            test_rmse = rmse(I2, T, Q, P)
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)

            """
            print("[Epoch %d/%d] train error: %f, test error: %f" %
                  (epoch + 1, self.num_epochs, train_rmse, test_rmse))
            """
        with open(PIKp, "w") as f:
            pickle.dump(P, f)

        with open(PIKq, "w") as f1:
            pickle.dump(Q, f1)

        return None


















