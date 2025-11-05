import torch
import random
import cv2
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam

def train(train_list,
          image_size,
          scale_ratio,
          n_bands,
          arch,
          model,
          optimizer,
          criterion,
          epoch,
          n_epochs):

    train_ref, train_lr, train_hr = train_list

    _, _, h, w = train_ref.shape
    h_str = random.randint(0, h-image_size-1)
    w_str = random.randint(0, w-image_size-1)

    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]
    train_hr = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    train_ref_numpy = train_ref.permute(0, 2, 3, 1).squeeze().numpy()
    train_lr = cv2.GaussianBlur(train_ref_numpy, (7, 7), 2)
    train_lr = cv2.resize(train_lr, (train_lr.shape[1] // scale_ratio, train_lr.shape[0] // scale_ratio))
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0)

    model.train()
    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()
    optimizer.zero_grad()

    out, out_stage1, gates = model(image_lr, image_hr)
    gate1, gate2, gate3 = gates

    para = 0.01
    loss_gate = (criterion["CVLoss"](gate1) +
                 criterion["CVLoss"](gate2) +
                 criterion["CVLoss"](gate3))

    loss_s1 = criterion["l1"](out, image_ref) 
    loss_s2 = criterion["l1"](out_stage1, image_ref) 

    loss = loss_s1 + loss_s2 + para * loss_gate
    loss.backward()
    optimizer.step()

    # Print log info
    print(f"Epoch [{epoch}/{n_epochs}] Loss:{loss.item():.4f}")


def validate(test_list, arch, scale_ratio, model):
   
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        out, _, _ = model(lr, hr)  

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out, scale_ratio)
        sam = calc_sam(ref, out)
  
    return rmse, psnr, ergas, sam