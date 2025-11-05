import os
from pathlib import Path
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from models.AELF import AELF
from utils import *
from metrics import *
from data_loader import build_datasets
from time import *
import args_parser
import cv2
from scipy.io import savemat

args = args_parser.args_parser()
print(args)

def main():
    args.root = './Datasets/'

    if args.dataset == 'Pavia':
        args.n_bands = 102
    elif args.dataset == 'Houston2018':
        args.n_bands = 48
    elif args.dataset == 'Chikusei':
        args.n_bands = 128
    # Custom dataloader
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)

    # Build the models
    if args.arch == 'AELF':
        model = AELF(
            hsi_c=args.n_bands,
            msi_c=args.n_select_bands,
            scale_factor=args.scale_ratio,
            n_feat=64,
            scale_unetfeats=32,
            kernel_size=3,
            bias=True,
        )
        
    # Load the trained model parameters
    model_path = args.model_path + '/' + args.arch + '/' + args.dataset + '×' + str(args.scale_ratio)
    model_path = model_path + '/' + '2025_11_05_14_34_01/net_9864epoch.pth' 

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach().cpu()  
    lr = test_lr.float().detach().to(device)
    hr = test_hr.float().detach().to(device)
    
    print()
    average_times = 10

    model.cuda()
    lr = lr.cuda()
    hr = hr.cuda()
    a_time = time()
    for i in range(average_times):
        if args.arch == 'AELF':
            out, _, _ = model(lr, hr)

    b_time = time()
    print('avarage times', average_times)
    print('test time(s)', (b_time - a_time) / average_times)
    print()

    print()
    print('Dataset:   {}'.format(args.dataset))
    print('Arch:   {}'.format(args.arch))
    print()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out, args.scale_ratio)
    sam = calc_sam(ref, out)
    cc = calc_cc(ref, out)
    ssim = calc_ssim(ref, out)
    uiqi = calc_uiqi(ref, out)

    print('RMSE:   {:.6f};'.format(rmse))
    print('PSNR:   {:.6f};'.format(psnr))
    print('ERGAS:   {:.6f};'.format(ergas))
    print('CC:   {:.6f};'.format(cc))
    print('SAM:   {:.6f};'.format(sam))
    print('SSIM:   {:.6f}.'.format(ssim))
    print('UIQI:   {:.6f}.'.format(uiqi))

    # bands order
    if args.dataset == 'Pavia':
        red, green, blue = 50, 30, 15
        vmax_sam, vmax_dif = 20, 0.5
    elif args.dataset == 'Chikusei':
        red, green, blue = 61, 41, 21
        vmax_sam, vmax_dif = 8, 0.5
    elif args.dataset == 'Houston2018':
        red, green, blue = 29, 19, 9
        vmax_sam, vmax_dif = 8, 0.3

    save_path = os.path.join('./figs', args.dataset + '×' + str(args.scale_ratio))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    def save_rgb_image_highres(tensor, red, green, blue, save_path_svg, resize_shape=None):
        img = np.squeeze(tensor)
        img_red = img[red, :, :][:, :, np.newaxis]
        img_green = img[green, :, :][:, :, np.newaxis]
        img_blue = img[blue, :, :][:, :, np.newaxis]
        img_rgb = np.concatenate((img_blue, img_green, img_red), axis=2)
        
        img_rgb = 255 * (img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb))
        img_rgb = (img_rgb).astype(np.uint8)

        if resize_shape:
            img_rgb = cv2.resize(img_rgb, resize_shape, interpolation=cv2.INTER_NEAREST)

        fig, ax = plt.subplots()
        ax.axis('off')  # no axes
        ax.imshow(img_rgb)
        plt.savefig(save_path_svg, format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

        return img_rgb
    
    def save_mat(data, save_path, key):
        savemat(save_path, {key: data})

    def save_difference_map(img1, img2, save_path, title='Difference',vmin=0, vmax=1):
        diff = 10 * np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        gray_diff = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_norm = gray_diff / 255.0

        plt.figure(figsize=(6, 5))
        im = plt.imshow(gray_norm, cmap='viridis', origin='upper',vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=30) 
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    def generate_sam_map(pred, ref, save_path, vmax=None, eps=1e-8, return_mean=True):
        dot = np.sum(pred * ref, axis=2)
        pred_norm = np.linalg.norm(pred, axis=2)
        ref_norm = np.linalg.norm(ref, axis=2)
        cos = dot / (pred_norm * ref_norm + eps)
        cos = np.clip(cos, -1, 1)
        angle_rad = np.arccos(cos)
        angle_deg = np.degrees(angle_rad)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(angle_deg, cmap='viridis', origin='upper', vmin=0, vmax=vmax)
        cbar = plt.colorbar(im, label='Spectral Angle (°)')
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Spectral Angle (°)', fontsize=14) 

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
        plt.close()

        if return_mean:
            return angle_deg, np.mean(angle_deg)
        else:
            return angle_deg 
                    
    # 1) Save LR
    lr = np.squeeze(test_lr.detach().cpu().numpy())
    lr_img = save_rgb_image_highres(lr, blue, green, red, 
                            os.path.join(save_path, f'{args.dataset}_lr.svg'),
                            resize_shape=(out.shape[2], out.shape[3]))
    # 2) Save Output
    out = np.squeeze(out)
    save_mat(out, os.path.join(save_path, f'{args.dataset}_{args.arch}_output.mat'), 'output')

    out_img = save_rgb_image_highres(out, blue, green, red, 
                            os.path.join(save_path, f'{args.dataset}_{args.arch}_out.svg'))

    # 3) Save ref & .mat 
    ref = np.squeeze(ref)
    save_mat(ref, os.path.join(save_path, f'{args.dataset}_ref.mat'), 'ref')

    ref_img = save_rgb_image_highres(ref, blue, green, red,
                            os.path.join(save_path, f'{args.dataset}_ref.svg'))

    # 4) Save diff & SAM
    save_difference_map(out_img, ref_img, 
                            os.path.join(save_path, f'{args.dataset}_{args.arch}_out_dif.svg'),
                            title='Out vs Ref Difference',vmax=vmax_dif)
    
    out_sam_map, out_sam_avg = generate_sam_map(
        out.transpose(1, 2, 0), ref.transpose(1, 2, 0),
        save_path=os.path.join(save_path, f'{args.dataset}_{args.arch}_out_sam.svg'),
        vmax=vmax_sam)
    
    print()
    print('sam_avg:   {:.6f};'.format(out_sam_avg))
    print('\n Test completed and all results saved!')


if __name__ == '__main__':
    main()
