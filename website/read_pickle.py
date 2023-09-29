import pickle
import base64
import io
from turtle import color
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from requests import patch
from torch.utils.data import DataLoader
import random
from scipy.interpolate import make_interp_spline
from static.LSANmodel.trainer import *
from static.LSANmodel.transformer import *
from static.LSANmodel.dataset import *
from static.LSANmodel.LSAN import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from scipy.signal import savgol_filter
def test():
    with open('pickles/rp.pickle','rb') as f:
    # with open('single_patient.pickle','rb') as f:
        data = pickle.load(f)
        visit = [data[0][0], data[0][0]]
        label = data[1]
        value = [data[3][0], data[3][0]]
        # dataset = Dataset(visit, label, value)
    fake_label = [[0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0]]
    dataset = Dataset(visit[0],fake_label, value[0])
    output_path = './result.txt'
    hidden = 256 
    layers = 8
    attn_heads = 8 
    dropout = 0.1
    batch_size = 12
    epochs = 10
    num_workers = 2
    with_cuda = 1
    lr = 0.0001
    adam_weight_decay = 0.01 ###
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    model_path = 'static/LSANmodel/model_parameters/0_51.6_models.pth'
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn,shuffle=False)
    embedding_dim = hidden

    test_model = LSAN(51, embedding_dim, transformer_hidden = hidden, attn_heads = attn_heads, transformer_dropout = dropout, transformer_layers = layers) 
    test_model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    test_model.eval()
    for unit in data_loader:
        padding_input, input_labels, value_input, x, y = unit
        y_pred = test_model(padding_input, value_input).squeeze(1)
        break
    rp_result = y_pred.tolist()
    time = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    fig = plt.figure(figsize = (12, 12))
    zs=savgol_filter(rp_result, 7, 3)
    plt.plot(time, rp_result, color='b', lw=1, alpha = 1)
    plt.plot(time, zs, color='r', lw=2)
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    test()
    # with open("static/patient1.pickle", 'rb') as f:
    #     data = pickle.load(f)
    # print(data)
    # count = 0
    # output_path = './result.txt'
    # hidden = 256 
    # layers = 8
    # attn_heads = 8 
    # dropout = 0.1
    # batch_size = 12
    # epochs = 10
    # num_workers = 2
    # with_cuda = 1
    # lr = 0.0001
    # adam_weight_decay = 0.01 ###
    # adam_beta1 = 0.9
    # adam_beta2 = 0.999
    # model_path = 'static/LSANmodel/model_parameters/0_51.6_models.pth'
    # embedding_dim = hidden
    # test_model = LSAN(51, embedding_dim, transformer_hidden = hidden, attn_heads = attn_heads, transformer_dropout = dropout, transformer_layers = layers) 
    # test_model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    # test_model.eval()
    # padding_input, input_labels, value_input, x, y = data
    # y_pred = test_model(padding_input, value_input).squeeze(1)
    # rp_result = y_pred.tolist()
    # time = time = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # fig = plt.figure(figsize = (12, 12))
    # zs=savgol_filter(rp_result, 7, 3)
    # plt.plot(time, rp_result, color='b', lw=1, alpha = 1)
    # plt.plot(time, zs, color='r', lw=2)
    # plt.title('Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Probability')
    # plt.ylim([0, 1])
    # plt.show()
