"""
Authors: Xiaoqing Liu
Data: 02/19/2022
Title: Gaussian NB with Gamma
Comments:

"""
import numpy as np
import pandas as pd

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
        grouped_data = data_with_label.groupby("label")
        lable_vals = np.sort(data_with_label["label"].unique())
        # groups = {}
        # mean_group = []
        # var_group = []

        mean_mat = data_with_label.groupby("label").mean()
        var_mat = data_with_label.groupby("label").var()

        # compute the prior
        labels_count = data_with_label["label"].value_counts().sort_index()
        prior_rate = np.array([i / sum(labels_count) for i in labels_count])

        # # compute mean, variance
        # for label_val in lable_vals:
        #     #     label_0 = test_s.get_group(label_val)
        #     groups[label_val] = grouped_data.get_group(label_val)
        #     mean_group.append(groups[label_val].values[:, :-1].mean(axis=0))
        #     var_group.append(groups[label_val].values[:, :-1].var(axis=0))

        return np.array(mean_mat), np.array(var_mat), prior_rate

    def fit_with_gamma(self, X_train, y_train):
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
        grouped_data = data_with_label.groupby("label")
        lable_vals = np.sort(data_with_label["label"].unique())
        groups = {}
        mean_group = []
        var_group = []

        mean_gamma = []
        var_gamma = []
        prior_gamma = []

        # compute the prior
        labels_count = data_with_label["label"].value_counts().sort_index()
        prior_rate = np.array([i / sum(labels_count) for i in labels_count])

        # compute mean, variance
        for label_val in lable_vals:
            #     label_0 = test_s.get_group(label_val)
            groups[label_val] = grouped_data.get_group(label_val)
            # mean_group.append(groups[label_val].values[:, :-1].mean(axis=0))
            # var_group.append(groups[label_val].values[:, :-1].var(axis=0))
            Cg, Rg, Wg = self.__calculate_parameter_with_k_gamma(groups[label_val].values[:, :-1],
                                                                 sum(labels_count))
            mean_gamma.append(Cg)
            var_gamma.append(Rg)
            prior_gamma.append(Wg)

        return np.array(mean_gamma), np.array(var_gamma), np.array(prior_gamma)

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