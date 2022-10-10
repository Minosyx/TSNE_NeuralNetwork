#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import argparse


def Hbeta(D=da.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = da.exp(-da.asarray(D) * beta)
    sumP = sum(P)
    H = da.log(sumP) + beta * da.sum(D * P) / sumP
    P = P / sumP
    
    return H, P


def x2p(X=da.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = da.shape(X)
    sum_X = da.sum(da.square(X), 1)
    D = da.add(da.add(-2 * da.dot(X, X.T), sum_X).T, sum_X) # || xi - xj || ^ 2
    P = da.zeros_like((n, n))
    beta = da.ones_like((n, 1))
    logU = da.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, da.concatenate((np.r_[0:i], np.r_[i+1:n]))]   # slice z pominieciem 0
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        print("LOL")
        while da.absolute(Hdiff) > tol and tries < 50:   # binary search

            print(tries)
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = da.asarray(beta[i])
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = da.asarray(beta[i])
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, da.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % da.mean(da.sqrt(1 / beta)))
    return P


def pca(X=da.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = da.shape(X)    # pobranie ksztaltu
    X = X - da.tile(da.mean(X, 0), (n, 1))  # Budowa macierzy odchylen
    (l, M) = np.linalg.eig(da.dot(X.T, X))  # Wyznaczenie wartosci i macierzy wektorow wlasnych z macierzy kowariancji
    Y = da.dot(X, M[:, 0:no_dims])  # Wyznaczenie wektorow wlasnych do wartosci wlasnych
    return Y


def tsne(X=da.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # X = pca(X, initial_dims).real  # wyłączenie PCA w celach testowych
    (n, d) = da.shape(X)
    initial_momentum = 0.5  # alfa(t) w gradiencie
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = da.random.normal(n, no_dims) # Y zainicjowane wartosciami z rozkladu normalnego
    dY = da.zeros_like((n, no_dims))
    iY = da.zeros_like((n, no_dims))
    gains = da.ones_like((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)    # znalezienie P
    P = P + da.transpose(P)
    P = P / da.sum(P)   # (p_{j|i} + p_{i|k})/2/n
    # P = P * 4.									# early exaggeration
    P = da.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = da.sum(da.square(Y), 1)
        num = -2. * da.dot(Y, Y.T)
        num = 1. / (1. + da.add(da.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / da.sum(num)
        Q = da.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = da.sum(da.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - da.tile(da.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = da.sum(P * da.log(P / Q))
            print(f"Iteration {iter + 1}: error is {C}")

        # Stop lying about P-values
        # if iter == 100:
        #     P = P / 4.

    # Return solution
    return Y


class FileTypeWithExtensionCheck(argparse.FileType):
    def __init__(self, mode='r', valid_extensions=None, **kwargs):
        super().__init__(mode, **kwargs)
        self.valid_extensions = valid_extensions

    def __call__(self, string):
        if self.valid_extensions:
            if not string.endswith(self.valid_extensions):
                raise argparse.ArgumentTypeError(
                    'Not a valid filename extension!')
        return super().__call__(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-SNE Algorithm')
    parser.add_argument('input_file', type=FileTypeWithExtensionCheck(valid_extensions=('txt','data')), help='Input file')
    parser.add_argument('-iter', type=int, default=1000, help='Number of iterations', required=False)
    parser.add_argument('-labels', type=FileTypeWithExtensionCheck(valid_extensions=('txt','data')), help='Labels file', required=False)
    parser.add_argument('-no_dims', type=int, help='Number of dimensions', required=True, default=2)
    parser.add_argument('-start_dims', type=int, help='Number of dimensions to start with after initial reduction using PCA', required=False, default=50)
    parser.add_argument('-perplexity', type=float, help='Perplexity of the Gaussian kernel', required=True, default=30.0)
    parser.add_argument('-exclude_cols', type=int, nargs='+', help='Columns to exclude', required=False)
    parser.add_argument('-step', type=int, help='Step between samples', required=False, default=1)
    # parser.add_argument('-max_rows', type=int, help='Number of rows to read', required=False)
    # parser.add_argument('-skip_rows', type=int, help='Number of rows to skip', default=0, required=False)

 
    
    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
    args = parser.parse_args()
    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels)
        args.labels.close()
        
    cols = None
    if args.exclude_cols:
        args.input_file.readline()
        last_pos = args.input_file.tell()
        ncols = len(args.input_file.readline().strip().split(' '))
        args.input_file.seek(last_pos)
        cols = np.arange(0, ncols, 1)
        cols = tuple(np.delete(cols, args.exclude_cols, axis=0))
    
    # X = np.loadtxt(args.input_file, usecols=cols, max_rows=args.max_rows, skiprows=args.skip_rows)
    X = np.loadtxt(args.input_file, usecols=cols)
    
    args.input_file.close()
    
    data = da.from_array(X[::args.step, :])

    means = da.mean(data, axis=0).compute()
    vars = da.var(data, axis=0).compute()
    # means = data.mean(axis=0)
    # vars = data.var(axis=0)
    
    for v in range(len(means)):
        print(f"Column {v} mean: {means[v]}, var: {vars[v]}")
    
    Y = tsne(data, args.no_dims, args.start_dims, args.perplexity, args.iter)
    
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.show()