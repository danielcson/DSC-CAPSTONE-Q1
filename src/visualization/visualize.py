import numpy as np

def cka(X,Y):
    # CKA formula from Kornblith et al.,(2019)

    # making a copy prevents modifying original arrays
    X = X.copy()
    Y = Y.copy()

    # center x and y first to do dot product
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # dot products
    x_xt = X.T.dot(X)
    y_yt = Y.T.dot(Y)
    x_yt = Y.T.dot(X)

    # Frobenius norm = root of the sum of squares of the entries when X and Y are centered
    return (x_yt**2).sum() / np.sqrt((x_xt**2).sum() * (y_yt**2).sum())
    # return np.linalg.norm(x=x_yt, ord='fro') / (np.linalg.norm(x=x_xt, ord='fro') * np.linalg.norm(x=y_yt, ord='fro'))

def plot_cka():
    #part b
    figure_5b = {'activation' : [],
    'kinematic_feature' : [],
    'cka' : []}

    #part c
    figure_5c = {'activation_1' : [],
    'activation_2' : [],
    'cka' : []}

    

