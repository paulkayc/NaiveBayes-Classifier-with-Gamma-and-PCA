"""
Authors: Xiaoqing Liu
Data: 02/13/2022
Title: HW01 PCA and Gaussian Naive Bayeswith Gamma
Comments:
  1.
  2.
"""

import numpy as np
import pandas as pd
import os
import time
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

class PCA_with_gamma(object):
    def __init__(self):
        pass

    def calculate_gamma_combine_chunk(self, file_path,chunksize,dims):
        gamma = np.zeros((dims, dims), dtype=float)
        tfr = pd.read_csv(file_path, chunksize=chunksize)

        for chunk in tfr:
            chunk_data = chunk.values[:, 1:]
            samples = chunk_data.shape[0]
            ones_row = np.ones((1, samples))
            chunk_data = np.r_[ones_row, chunk_data.T]
            gamma += np.dot(chunk_data, chunk_data.T)
        # Q = gamma[1:-1, 1:-1]
        # L = gamma[1:-1, 0]
        # counts = gamma[0, 0]
        return gamma

    def calculate_gamma_z_score_data(self, file_path, z_score_file, chunksize, gamma):
        """
        This function is to z-score the input data x' = (x-mu)/sigma, and get the gamma after z-score the input
        """
        Q = gamma[1:-1, 1:-1]
        L = gamma[1:-1, 0]
        counts = gamma[0, 0]
        Q_diag = np.diag(Q)
        mu = L / counts
        sigma = np.sqrt(Q_diag/counts - (L*L / (counts ** 2)))
        """
        # Test the z-score transform with sklearn
        # data_x_z_score_std, mean_std, var_std = self.get_std_z_score_data(file_path)
        # 
        # print("The different of mean value is:", np.sum(np.abs(mu - mean_std)))
        # print("The different of sqrt value is:", np.sum(np.abs(sigma - np.sqrt(var_std))))
        """

        gamma_z_scored = np.zeros(gamma.shape, dtype=float)
        tfr = pd.read_csv(file_path, chunksize=chunksize)
        header = True
        for chunk in tfr:
            # z-score the input data
            z_score_data_x = (chunk.values[:, 1:-1] - mu) / sigma
            z_score_data = np.c_[chunk.values[:,0], z_score_data_x, chunk.values[:,-1]]
            z_score_data = pd.DataFrame(z_score_data, index=None, columns=chunk.columns)
            # data_x_z_score_std, std_mean, std_var = self.get_z_score_chunk(chunk.values[:, 1:-1])
            # write z-scored data into csv file

            z_score_data.to_csv(z_score_file, index=False, header=header, mode='a')
            header = False

            # get the gamma after z-score of the data
            chunk_data = z_score_data.values[:, 1:]
            samples = z_score_data.shape[0]
            ones_row = np.ones((1, samples))
            chunk_data = np.r_[ones_row, chunk_data.T]
            gamma_z_scored += np.dot(chunk_data, chunk_data.T)

        return gamma_z_scored

    def get_z_score_chunk(self, chunk_data):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(chunk_data)
        data_x_z_score_std = scaler.transform(chunk_data)
        return data_x_z_score_std, scaler.mean_, scaler.var_


    def get_std_z_score_data(self, file_path):
        from sklearn.preprocessing import StandardScaler
        tfr = pd.read_csv(file_path, chunksize=10000, iterator=True)
        df = pd.concat(tfr, ignore_index=True)
        data = df.values
        data_x = data[:, 1:-1]
        scaler = StandardScaler()
        scaler.fit(data_x)
        data_x_z_score_std = scaler.transform(data_x)
        return data_x_z_score_std, scaler.mean_, scaler.var_

    def compute_correlation_matrix(self, gamma):
        dims = gamma.shape[0]
        dims -= 2
        corrcoef_matrix = np.zeros((dims, dims))
        Q = gamma[1:-1, 1:-1]
        L = gamma[1:-1, 0]
        counts = gamma[0, 0]
        for i in range(dims):
            for j in range(dims):
                corrcoef_matrix[i, j] = (counts * Q[i, j] - L[i] * L[j]) / (
                        np.sqrt(counts * Q[i, i] - L[i] ** 2) * np.sqrt(counts * Q[j, j] - L[j] ** 2))

        return corrcoef_matrix

    def get_pc_num_eigvects(self, corrcoef_matrix, n_compoment, percentage):
        eigvals, eigvects = np.linalg.eig(corrcoef_matrix)

        if n_compoment is not None:
            pc_num = n_compoment
        else:
            pc_num = self.percentage2pcnum(eigvals, percentage)

        eigval_index = np.argsort(eigvals)
        max_n_eigval_index = eigval_index[-1:-(pc_num + 1):-1]
        pc_num_eigvects = eigvects[:, max_n_eigval_index]

        # compute eigval and eigvects with SVD
        u, d, v = np.linalg.svd(corrcoef_matrix)
        if n_compoment is not None:
            pc_num = n_compoment
        else:
            pc_num = self.percentage2pcnum(d, percentage)
        pc_num_eigvects = u[:, :pc_num]

        select_percentage = self.pcnum2percentage(eigvals, pc_num)

        return pc_num_eigvects, select_percentage

    # According percentage to decide the number of PC
    def percentage2pcnum(self, eigvals, percentage):
        sort_eigvals = np.sort(eigvals)
        sort_eigvals = sort_eigvals[-1::-1]
        total_eigvals = np.sum(sort_eigvals)
        tmp_sum = 0
        pcnum = 0
        for i in sort_eigvals:
            tmp_sum += i
            pcnum += 1
            if tmp_sum >= total_eigvals * percentage:
                return pcnum

    def pcnum2percentage(self, eigvals, n_compoment):
        sort_eigvals = np.sort(eigvals)
        sort_eigvals = sort_eigvals[-1::-1]
        return np.sum(sort_eigvals[0:n_compoment]) / np.sum(sort_eigvals)



    def pca_fit_with_gamma(self, file_path, z_score_file, chunksize, dims, n_component=None, percentage=0.99):
        """

        """
        start_corrcoef_gamma = time.time()

        gamma_original = self.calculate_gamma_combine_chunk(file_path, chunksize, dims)
        gamma = self.calculate_gamma_z_score_data(file_path, z_score_file, chunksize, gamma_original)
        corrcoef_matrix = self.compute_correlation_matrix(gamma)
        n_component_eigvects, select_percentage = self.get_pc_num_eigvects(corrcoef_matrix,
                                                                           n_component,
                                                                           percentage)

        end_corrcoef_gamma = time.time()
        print("Time for PCA with Gamma is:", end_corrcoef_gamma - start_corrcoef_gamma)


        return n_component_eigvects, select_percentage

    def pca_transform(self, n_component_eigvects, data):
        return np.dot(data, n_component_eigvects)

class Gaussian_NaiveBayes(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        """
        This function is to calculate the mean, variance, prior on the training data
        Input:
        -- X_train:
        -- y_train:
        Output:
        -- mean
        -- variance
        -- prior
        """
        label_names = []
        for i in range(X_train.shape[1]):
            label_names.append(f"x{i}")
        label_names.append('label')
        data_with_label = np.c_[X_train, y_train]

        data_with_label = pd.DataFrame(data_with_label, columns=label_names)
        # grouped_data = data_with_label.groupby("label")
        # lable_vals = np.sort(data_with_label["label"].unique())

        mean_mat = data_with_label.groupby("label").mean()
        var_mat = data_with_label.groupby("label").var()

        # compute the prior
        labels_count = data_with_label["label"].value_counts().sort_index()
        prior_rate = np.array([i / sum(labels_count) for i in labels_count])

        return np.array(mean_mat), np.array(var_mat), prior_rate



    def fit_with_gamma_new(self, X_train, y_train, n_class):
        """
        This function is to calculate the mean, variance, prior on the training data
        Input:
        -- X_train:
        -- y_train:
        Output:
        -- mean
        -- variance
        -- prior
        """
        labels = np.arange(n_class)
        data_with_labels = np.c_[X_train, y_train]
        total_counts = X_train.shape[0]
        mean_gamma = []
        var_gamma = []
        prior_gamma = []

        for label in labels:
            data_g = data_with_labels[data_with_labels[:,-1]==label]
            Cg, Rg, Wg = self.__calculate_parameter_with_k_gamma(data_g[:, :-1],
                                                                 total_counts)
            mean_gamma.append(Cg)
            var_gamma.append(Rg)
            prior_gamma.append(Wg)

        return np.array(mean_gamma), np.array(var_gamma), np.array(prior_gamma)

    def __calculate_parameter_with_k_gamma(self, grouped_data, total_n):
        """
        This function is to calculate the mean, variance, prior from gamma of class k
        Input:
        -- grouped_data: the dataset with class k, one row one sample
        Output:
        -- mean_gamma: Q the dataset
        -- var_gamma: L of the dataset
        -- Ng: N of the dataset
        """
        ones_add = np.ones((1,grouped_data.shape[0]))
        data_z = np.r_[ones_add, grouped_data.T]

        result = np.dot(data_z, data_z.T)

        Ng = result[0, 0]
        Lg = result[1:, 0]

        Q_matrix = result[1:, 1:]
        Qg = np.diag(Q_matrix)
        var_gamma = Qg / Ng - (Lg * Lg) / (Ng ** 2)
        mean_gamma = Lg / Ng
        prior_gamma = Ng/total_n

        return mean_gamma, var_gamma, prior_gamma

    def __gaussian_possibility(self, mu, var, one_sample):
        """
        This function is to implement the formula:
        P(x|Ck) = 1/sqrt(2*pi*var) * exp(-(x-mu)**2 /(2*var))
       Input:
        -- mu: mean
        -- var: variance
        -- one_sample: one sample features data
        Output:
        -- gaussian_p: Gaussian possibility
        """
        gaussian_p_mat = 1 / np.sqrt(2 * np.pi * var) * np.exp(-(one_sample - mu) ** 2 / (2 * var))
        gaussian_p = pd.DataFrame(gaussian_p_mat).prod(axis=1)
        return gaussian_p

    def predict(self, mean, var, prior, data_x):
        """
        This function is to prect the class of the input data, with the knowledge of mean, variance, prior:
       Input:
        -- mu: mean
        -- var: variance
        -- prior: prior possibility of each class
        -- data_x: input data_x, one row one samsple
        Output:
        -- class: Gaussian possibility
        """
        pred_possiblity = [self.__gaussian_possibility(mean, var, row) * prior for row in data_x]
        pred_class = np.argmax(pred_possiblity, axis=1)
        return pred_class


def main():
    file_path = "cancer_100k.csv"
    z_score_file = "z_score_dataset.csv"
    pca_file = "pca_dataset.csv"

    if os.path.exists(z_score_file):
        os.remove(z_score_file)
    if os.path.exists(pca_file):
        os.remove(pca_file)
    
    chunksize = 10000
    dims = 32
    n_class = 2
    n_component = 8
    pca = PCA_with_gamma()
    n_component_eigvects, percentage = pca.pca_fit_with_gamma(file_path,
                                                              z_score_file,
                                                              chunksize,
                                                              dims,
                                                              n_component=n_component)
    print("{} components contain {} main factors".format(n_component, percentage))

    """
     Transform the z_score data per chunk, save the dimention reduction data to pca_file 
    """
    tfr = pd.read_csv(z_score_file, chunksize=chunksize)
    columns = []
    columns.append("patient_id")
    for i in range(n_component):
        columns.append(f"d{i}")
    columns.append('label')

    header = True
    for chunk in tfr:
        chunk_data_x = chunk.values[:, 1:-1]
        chunk_data_reduction = pca.pca_transform(n_component_eigvects, chunk_data_x)
        chunk_data_reduction = np.c_[chunk.values[:, 0], chunk_data_reduction, chunk.values[:, -1]]

        chunk_data_reduction = pd.DataFrame(chunk_data_reduction, index=None, columns=columns)
        chunk_data_reduction.to_csv(pca_file, index=False, header=header, mode='a')
        header = False


    """For Gaussion Naive Bayes part"""
    pca_data = pd.read_csv(pca_file)
    data_transform_pca = pca_data.values[:,1:-1]
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(data_transform_pca),
                                                        pd.DataFrame(pca_data.values[:, -1]),
                                                        test_size=0.4,
                                                        random_state=1)

    NB_std = GaussianNB()
    NB_std.fit(X_train, y_train.values.ravel())

    y_train_predict = NB_std.predict(X_train)
    y_test_predict = NB_std.predict(X_test)

    print("Use sklearn the accuracy on train set is:", accuracy_score(y_train, y_train_predict))

    print("Use sklearn the accuracy on test set is:", accuracy_score(y_test, y_test_predict))

    NB_my = Gaussian_NaiveBayes()
    # mu, var, prior = NB_my.fit(X_train, y_train)
    # mu, var, prior = NB_my.fit_with_gamma(X_train, y_train)
    mu, var, prior = NB_my.fit_with_gamma_new(X_train, y_train, n_class)
    y_train_predict_my = NB_my.predict(mu, var, prior, X_train.values)
    y_test_predict_my = NB_my.predict(mu, var, prior, X_test.values)

    print("Use MY the accuracy on train set is:", accuracy_score(y_train, y_train_predict_my))

    print("Use MY the accuracy on test set is:", accuracy_score(y_test, y_test_predict_my))






if __name__ == '__main__':
    main()