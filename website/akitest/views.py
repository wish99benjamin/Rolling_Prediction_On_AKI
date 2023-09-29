import json
import io
from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import User
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.contrib import auth
from django.http import HttpResponseRedirect
import torch
import os
import numpy as np
import pickle
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from scipy.interpolate import make_interp_spline
from static.LSANmodel.trainer import *
from static.LSANmodel.transformer import *
from static.LSANmodel.dataset import *
from static.LSANmodel.LSAN import *
from scipy.interpolate import make_interp_spline
import warnings
import base64
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
def startpage(req):
    return render(req, 'startPage.html')

@login_required
def homepage(req):
    return render(req, 'homepage.html')

def login(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('/homepage/')

    username = request.POST.get('username', '')
    password = request.POST.get('password', '')

    user = auth.authenticate(username=username, password=password)

    if user is not None and user.is_active:
        auth.login(request, user)
        return HttpResponseRedirect('/homepage/')
    else:
        return render(request, "login.html")


def logout(request):
    auth.logout(request)
    return render(request, "logout.html")

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = User.objects.create_user(username=username, password=password)
        user.save()
        messages.success(request, "註冊成功!")
        return HttpResponseRedirect('/login/')
    else:
        return render(request, "register.html")

@login_required
def predictfile(req):
    return render(req, 'predictFile.html')

@login_required
def predictresultforfile(request):
    a = 0
    if request.method == 'POST':
        if 'Upload' in request.POST:
            state = 'Upload'
            for filename in os.listdir("./static/media"):
                file = "./static/media/" + filename
                os.remove(file)
            try:
                UploadFile = request.FILES['document']
                fs = FileSystemStorage()
                fs.save('./static/media/File', UploadFile)
            except:
                messages.success(request, "上傳失敗，請重新上傳檔案")
                return HttpResponseRedirect('/homepage/')
    # try:
    with open('./static/media/File', 'rb') as f:
        patient = pickle.load(f)
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
    model_path = './static/LSANmodel/model_parameters/0_51.6_models.pth'
    embedding_dim = hidden
    test_model = LSAN(51, embedding_dim, transformer_hidden = hidden, attn_heads = attn_heads, transformer_dropout = dropout, transformer_layers = layers) 
    test_model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    test_model.eval()
    padding_input, input_labels, value_input, x, y = patient
    y_pred = test_model(padding_input, value_input).squeeze(1)
    rp_result = y_pred.tolist()
    time = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    image_path = "./static/media/AKIresult.png"
    zs=savgol_filter(rp_result, 7, 3)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(time, rp_result, color='b', lw=1, alpha = 1)
    plt.plot(time, zs, color='r', lw=1.5)
    plt.title('Prediction')
    plt.xlabel('Hours')
    plt.ylabel('Probability')
    plt.ylim([0, 1])
    plt.savefig(image_path, format='png')
    plt.close()      
    ctx = {
        "label" : 0,
    } 
    if rp_result[11]>=0.5:
            ctx["label"] = 1
    print(ctx["label"])
    return render(request, "PredictResultForFile.html", ctx)
    # except:
        # messages.success(request, "檔案預測失敗，請重新上傳檔案")
        # return HttpResponseRedirect('/homepage/')


