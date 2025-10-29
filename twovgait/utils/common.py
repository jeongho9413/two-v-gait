import matplotlib.pyplot as plt
from matplotlib import cm
# from sklearn import neighbors
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F


def t_sne(features_g, labels_g, text):
    #features_g /= np.linalg.norm(features_g, ord=2, axis=1)[:, np.newaxis]
    embedding = TSNE(n_components=2).fit_transform(features_g)
    """
    labels_g_enc = np.full((labels_g.shape[0], ), 0, dtype='int16')
    for i in range(labels_g.shape[0]):
        for j in range(labels_g.shape[1]):
            if labels_g[i][j] == 0:
                continue
            elif labels_g[i][j] == 1:
                labels_g_enc[i] = j+1
    """
    
    plt.close()
    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(1,1,1)
    #plt.scatter(embedding[:,0], embedding[:,1], c=labels_g_enc, cmap=cm.tab20)
    plt.scatter(embedding[:,0], embedding[:,1], c=labels_g, cmap=cm.tab20)
    ax.set_aspect('equal')
    plt.colorbar()
    fig.savefig('./t-sne_{' + text + '}.png')


def extract_feature(net, test_x_dep_side, test_x_dep_back, test_x_speed, test_y, device, batch_mini):
    batch_total = test_y.size(0)
    batch_num = batch_total//batch_mini
    
    pred_x = None
    for batch_idx in range(batch_num):
        if batch_idx == batch_num - 1:
            pred_x_sample = net(test_x_dep_side[(batch_idx*batch_mini):batch_total].to(device), test_x_dep_back[(batch_idx*batch_mini):batch_total].to(device), test_x_speed[(batch_idx*batch_mini):batch_total].to(device))
        elif batch_idx != batch_num:
            pred_x_sample = net(test_x_dep_side[(batch_idx*batch_mini):((batch_idx+1)*batch_mini)].to(device), test_x_dep_back[(batch_idx*batch_mini):((batch_idx+1)*batch_mini)].to(device), test_x_speed[(batch_idx*batch_mini):((batch_idx+1)*batch_mini)].to(device))
        if pred_x != None:
            pred_x = torch.cat((pred_x, pred_x_sample), dim=0)
        elif pred_x == None:
            pred_x = pred_x_sample
            
    pred_x = pred_x.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()
        
    return pred_x, test_y