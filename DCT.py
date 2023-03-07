import numpy as np
from PIL import Image
import random
import copy
import matplotlib.pylab as plt
from math import log10, sqrt
from scipy.fftpack import dct,idct
## this is the code get from https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
from zigzag import zigzag, inverse_zigzag
img1 = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\airplane.png").convert('L')
#img1 = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\baboon.png").convert('L')
def dct_2d(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')
def idct_2d(matrix):
    return idct(idct(matrix.T, norm='ortho').T, norm='ortho') 
def get_window(img,i,j):
	return img[i:i+8,j:j+8]
def PSNR(original, target):
    mse = np.mean((original - target) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def DCT_Compression(img,n_c):
    img_gt = np.asarray(img)
    img = np.asarray(img)
    
    H = img.shape[0]
    W = img.shape[1]
    for i in range(0,H,8):
        for j in range(0,W,8):
            block = get_window(img,i,j)
            C_M = dct_2d(block)
            C_V = zigzag(C_M)
            Mask_1 = np.ones(n_c)
            Mask_2 = np.zeros(64-n_c)
            Mask = np.concatenate((Mask_1,Mask_2))
            C_V_mask = np.multiply(C_V,Mask)
            C_M_rec = inverse_zigzag(C_V_mask,8,8)
            block_rec = idct_2d(C_M_rec)
            img[i:i+8,j:j+8] = block_rec
    psnr = PSNR(img_gt,img)
    if n_c == 2 or n_c == 4 or n_c == 8 or n_c == 16 or n_c == 32:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img.save('img1_{}.png'.format(n_c))
    return psnr

psnr_l = []
for i in range(2,64):
    psnr = DCT_Compression(img1,i)
    psnr_l.append(psnr)
    if i == 2 or i == 4 or i == 8 or i == 16 or i == 32:
        print(i,psnr)

y = np.asarray(psnr_l)
x = range(2,64,1)
x = np.asarray(x)
plt.plot(x, y)
plt.xlabel('K') 
plt.ylabel('PSNR') 
plt.title('Reconstruction PSNR for differnet K, airplane.png')
plt.show()