import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rsatoolbox

import sys
# sys.path.append('/home/dson')
sys.path.append('/Users/danielson/')

from dsc_capstone_q1.src.model.train_agent import extract_kinematic_activations
# LOADED_HOOK_DICT, TOTAL_KINEMATIC_DICT = extract_kinematic_activations()

from dsc_capstone_q1.src.model.model_utils import load_hook_dict


# error bc its being paralleized?
# https://stackoverflow.com/questions/46607973/error-in-python-free-invalid-pointer-0x00007fc3c90dc98e
LOADED_HOOK_DICT = load_hook_dict('test/hook_dict.npy')
TOTAL_KINEMATIC_DICT = load_hook_dict('test/kinematic_dict.npy')


def cka(X,Y):
    """
    Implementation of CKA similarity index as formulated by Kornblith et al.,(2019)
    """

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


def plot_cka_5b():
    """
    plot cka similarity matrix between activations and kinematic features
    """
    loaded_hook_dict, total_kinematic_dict = LOADED_HOOK_DICT, TOTAL_KINEMATIC_DICT
    #part b
    figure_5b = {'activation' : [],
    'kinematic_feature' : [],
    'cka' : []}

    for i in total_kinematic_dict:
        total_kinematic_dict[i] = np.array(total_kinematic_dict[i])

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

    df_b = pd.DataFrame(figure_5b).drop_duplicates().pivot('kinematic_feature', 'activation', 'cka')
    plot_b = sns.heatmap(df_b, cbar_kws={'label':'Feature encoding (CKA)'}, cmap="Blues")
    fig_name = "cka_activation_vs_kinematic.png"
    plt.savefig("outputs/cka_activation_vs_kinematic.png")
    print(fig_name + ' has been saved to outputs folder')

    return plot_b

def plot_cka_5c():
    """
    plot cka similarity matrix between activation layers
    """
    loaded_hook_dict = LOADED_HOOK_DICT
    #part c
    figure_5c = {'activation_1' : [],
    'activation_2' : [],
    'cka' : []}

    # get combinations between activations
    for activation1 in loaded_hook_dict.keys():
        for activation2 in loaded_hook_dict.keys():
            cka_calc = cka(loaded_hook_dict[activation1], loaded_hook_dict[activation2])
            if activation1 == activation2:
                cka_calc = 1
            figure_5c['cka'].append(cka_calc)
            figure_5c['activation_1'].append(activation1)
            figure_5c['activation_2'].append(activation2)

    df_c = pd.DataFrame(figure_5c).pivot('activation_1', 'activation_2', 'cka')
    plot_c = sns.heatmap(df_c, cbar_kws={'label':'Representational similarity (CKA)'}, cmap="Blues")
    plt.savefig("outputs/cka_activations.png")
    print("cka_activations.png has been saved to outputs folder")
    return plot_c

def plot_rsa_5a(activation,kinematic):
    """
    plot rsa similarity between kinematic feature and activation layer
    activation: 'mean_linear' or 'log_std_linear'
    kinematic: 'joint_angles', 'joint_velocities', 'actuator_forces'
    """
    total_kinematic_dict = TOTAL_KINEMATIC_DICT
    loaded_hook_dict = LOADED_HOOK_DICT
    kmeans = KMeans(n_clusters=50, random_state=0).fit(total_kinematic_dict[kinematic])
    df = pd.DataFrame(total_kinematic_dict[kinematic])
    df['cluster_label'] = pd.Series(kmeans.labels_)

    all_calcs = []
    b = loaded_hook_dict[activation]

    for i in range(50):
      a = df[df['cluster_label']==i].drop(['cluster_label'], axis=1).values
      c = list(np.dot(a,b.T))
      all_calcs += c

    out = np.array(all_calcs)
    data = rsatoolbox.data.Dataset(out)
    rdms = rsatoolbox.rdm.calc_rdm(data)
    title = activation + ' vs. ' + kinematic
    rsatoolbox.vis.show_rdm(rdms, rdm_descriptor=title, show_colorbar='panel', figsize=(8,8))
    plt.savefig("outputs/rsa.png")
    print('rsa.png has been saved to outputs folder')