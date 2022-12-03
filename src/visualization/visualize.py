import numpy as np
import pandas as pd
import seaborn as sns
from dsc_capstone_q1.src.model.train_agent import extract_kinematic_activations

loaded_hook_dict, total_kinematic_dict = extract_kinematic_activations()

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

    # nested loop through each combination and add to the dictionary
    # if its geom_positions, reshape it from 3d to 2d

    # get combinations between kinematics and activations
    for feat in total_kinematic_dict.keys():
        for activation in loaded_hook_dict.keys():
            if feat == 'geom_positions':
                x_3d = total_kinematic_dict[feat]
                nsamples, nx, ny = x_3d.shape
                x = x_3d.reshape((nsamples,nx*ny))
                cka_calc = cka(loaded_hook_dict[activation], x)

                figure_5b['activation'].append(activation)
                figure_5b['kinematic_feature'].append(feat)
                figure_5b['cka'].append(cka_calc)
            else:
                cka_calc = cka(loaded_hook_dict[activation], total_kinematic_dict[feat])

                figure_5b['activation'].append(activation)
                figure_5b['kinematic_feature'].append(feat)
                figure_5b['cka'].append(cka_calc)


    # get combinations between activations
    for activation1 in loaded_hook_dict.keys():
        for activation2 in loaded_hook_dict.keys():
            cka_calc = cka(loaded_hook_dict[activation1], loaded_hook_dict[activation2])
            if activation1 == activation2:
                cka_calc = 1
            figure_5c['cka'].append(cka_calc)
            figure_5c['activation_1'].append(activation1)
            figure_5c['activation_2'].append(activation2)

    df_b = pd.DataFrame(figure_5b).drop_duplicates().pivot('kinematic_feature', 'activation', 'cka')
    sns.heatmap(df_b, cbar_kws={'label':'Feature encoding (CKA)'}, cmap="Blues")

    df_c = pd.DataFrame(figure_5c).pivot('activation_1', 'activation_2', 'cka')
    sns.heatmap(df_c, cbar_kws={'label':'Representational similarity (CKA)'}, cmap="Blues")
