from torch.nn import functional as F
import cv2
import torch
import numpy as np
import scipy.io as scio
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import savemat

def load_srf_from_txt(filepath):
    df = pd.read_excel(filepath, header=None).dropna()
    wavelengths_nm = df.iloc[:, 0].values
    srf_list = []
    for i in range(2, df.shape[1]): # pan:1; msi:2-5
        srf_list.append({
            "wavelengths": wavelengths_nm,  
            "responses": df.iloc[:, i].values
        })
    return srf_list

def generate_MSI(hsi_cube, hsi_wavelengths, srf_list):
    H, W, B = hsi_cube.shape
    N = len(srf_list)
    msi_cube = np.zeros((H, W, N))

    for i, srf in enumerate(srf_list):
        interp_func = interp1d(srf["wavelengths"], srf["responses"],
                               kind="linear", bounds_error=False, fill_value=0)
        weights = interp_func(hsi_wavelengths)

        if np.sum(weights) == 0:
            print(f" There is no overlap between the SRF and HSI bands of channel {i}! Return to 0")
            continue

        weights /= np.sum(weights)
        msi_cube[:, :, i] = np.tensordot(hsi_cube, weights, axes=([2], [0]))

    return msi_cube

def build_datasets(root, dataset, size, n_select_bands, scale_ratio):
    if dataset == 'Pavia':
        img = scio.loadmat(root + '/' + 'Pavia.mat')['pavia']*1.0
        wavenumbers = np.linspace(430, 860, 102) 
    elif dataset == 'Houston2018':
        img = scio.loadmat(root + '/' + 'Houston2018.mat')['Houston2018']*1.0
        wavenumbers = np.linspace(380, 1050, 48)
    elif dataset == 'Chikusei':
        img = scio.loadmat(root + '/' + 'Chikusei.mat')['Chikusei']*1.0
        wavenumbers = np.linspace(343, 1018, 128)

    srf_list = load_srf_from_txt('./Datasets/GF-2-PMS-1.xls')

    print (img.shape)
    max = np.max(img)
    min = np.min(img)
    img = 255*((img - min) / (max - min + 0.0))

    # throwing up the edge
    w_edge = img.shape[0]//scale_ratio*scale_ratio-img.shape[0]
    h_edge = img.shape[1]//scale_ratio*scale_ratio-img.shape[1]
    w_edge = -1  if w_edge==0  else  w_edge
    h_edge = -1  if h_edge==0  else  h_edge
    img = img[:w_edge, :h_edge, :]

    # cropping area
    width, height, n_bands = img.shape 
    w_str = (width - size) // 2 
    h_str = (height - size) // 2 
    w_end = w_str + size
    h_end = h_str + size
    img_copy = img.copy()

    # test sample
    gap_bands = n_bands / (n_select_bands-1.0)
    test_ref = img_copy[w_str:w_end, h_str:h_end, :].copy()
    test_lr = cv2.GaussianBlur(test_ref, (7,7), 2)
    test_lr = cv2.resize(test_lr, (size//scale_ratio, size//scale_ratio))

    test_hr = generate_MSI(test_ref, wavenumbers, srf_list)

    # training sample
    img[w_str:w_end,h_str:h_end,:] = 0
    train_ref = img
    train_lr = cv2.GaussianBlur(train_ref, (7,7), 2)
    train_lr = cv2.resize(train_lr, (train_lr.shape[1]//scale_ratio, train_lr.shape[0]//scale_ratio))
    
    train_hr = generate_MSI(train_ref, wavenumbers, srf_list)
    
    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0) 
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0) 
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0) 
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0) 
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0) 

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]
