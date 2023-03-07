import numpy as np
from PIL import Image
import random
import copy
from math import log10, sqrt
#img1 = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\airplane.png").convert('L')
img1 = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\baboon.png").convert('L')
img_l = []
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\airplane.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\zelda.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\baboon.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\barbara.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\boat.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\tulips.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\girl.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\goldhill.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\peppers.png").convert('L')
img_l.append(img)
img = Image.open("C:\\Users\\Yuning Huang\\Desktop\\Study\\ECE634\\project2\\sample_image\\monarch.png").convert('L')
img_l.append(img)


def get_window(img,i,j):
	return img[i:i+4,j:j+4]
def PSNR(original, target):
    mse = np.mean((original - target) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def VQ(img,level):
	img = np.asarray(img)
	H = img.shape[0]
	W = img.shape[1]
	count = 0
	block_l = []
	pattern_l = []
	for i in range(0,H,4):
		for j in range(0,W,4):
			block = get_window(img,i,j)
			block_l.append(block)
			random_number = random.randint(1, 10)
			if count<level:
				flag = True
				for k in range(count):
					if (block == block_l[k]).all():
						flag = False
						print("false")
						break
				#random_number = random.randint(1, 100)
				#if random_number <= 20:
				#	if flag == True:
				#		pattern_l.append(block)
				#		count = count + 1
				pattern_l.append((256//level)*count*np.ones((4,4)))
				count = count + 1
	block_l = np.asarray(block_l)
	pattern_l = np.asarray(pattern_l)
	a = pattern_l[0]
	old_pattern = copy.deepcopy(pattern_l)
	look_table = []
	for k in range(block_l.shape[0]):
		block = block_l[k]
		min = float('inf')
		selected_l = 0
		for l in range(level):
			pattern = pattern_l[l]
			distance = np.linalg.norm(block - pattern)
			if distance <= min:
				selected_l = l
				min = distance
		look_table.append(selected_l)
	d_l = []
	sum = 0
	for l in range(level):
		for k in range(block_l.shape[0]):
			if look_table[k] == l:
				sum = sum + np.linalg.norm(block_l[k]-pattern_l[l])
	distortion0 = sum / block_l.shape[0]
	print("check_point0",distortion0)
	d_l.append(distortion0)
	for q in range(10):
		for l in range(level):
			count = 0
			sum = np.zeros((4,4))
			for k in range(block_l.shape[0]):
				if look_table[k] == l:
					count = count + 1
					sum = sum + block_l[k]
			#if count == 0:
				#print(l,"count=0!")
			new_pattern = sum/count
			pattern_l[l] = new_pattern
		#sum = 0
		#for l in range(level):
		#	for k in range(block_l.shape[0]):
		#		if look_table[k] == l:
		#			sum = sum + np.linalg.norm(block_l[k]-pattern_l[l])
		#distortion1 = sum / block_l.shape[0]
		#print("check_point1",distortion1)
		look_table = []
		for k in range(block_l.shape[0]):
			block = block_l[k]
			min = float('inf')
			selected_l = 0
			for l in range(level):
				pattern = pattern_l[l]
				distance = np.linalg.norm(block - pattern)
				if distance <= min:
					selected_l = l
					min = distance
			look_table.append(selected_l)

		sum = 0
		for l in range(level):
			for k in range(block_l.shape[0]):
				if look_table[k] == l:
					sum = sum + np.linalg.norm(block_l[k]-pattern_l[l])
		distortion1 = sum / block_l.shape[0]
		print(distortion1)
		d_l.append(distortion1)
	dl = np.asarray(d_l)
	np.save('dl.npy',dl)
	np.save('test.npy',pattern_l)
	k = 0
	for i in range(0,H,4):
		for j in range(0,W,4):
			img[i:i+4,j:j+4] = pattern_l[look_table[k]]
			k = k + 1
	img = Image.fromarray(img.astype('uint8'), 'L')
	return img


def VQ_M_train(img_l,level):
	pattern_l = []
	for k in range(level):
		pattern_l.append((256//level)*k*np.ones((4,4)))
	block_l = []
	for k in range(len(img_l)):
		img = np.asarray(img_l[k])
		H = img.shape[0]
		W = img.shape[1]
		
		for i in range(0,H,4):
			for j in range(0,W,4):
				block = get_window(img,i,j)
				if block.shape[0]!=4 or block.shape[1]!=4:
					print(k)
					print("wrong!")
				block_l.append(block)

	block_l = np.asarray(block_l)
	print(block_l.shape)
	pattern_l = np.asarray(pattern_l)
	a = pattern_l[0]
	old_pattern = copy.deepcopy(pattern_l)
	look_table = []
	for k in range(block_l.shape[0]):
		block = block_l[k]
		min = float('inf')
		selected_l = 0
		for l in range(level):
			pattern = pattern_l[l]
			distance = np.linalg.norm(block - pattern)
			if distance <= min:
				selected_l = l
				min = distance
		look_table.append(selected_l)
	d_l = []
	sum = 0
	for l in range(level):
		for k in range(block_l.shape[0]):
			if look_table[k] == l:
				sum = sum + np.linalg.norm(block_l[k]-pattern_l[l])
	distortion0 = sum / block_l.shape[0]
	print("check_point0",distortion0)
	d_l.append(distortion0)
	for q in range(10):
		for l in range(level):
			count = 0
			sum = np.zeros((4,4))
			for k in range(block_l.shape[0]):
				if look_table[k] == l:
					count = count + 1
					sum = sum + block_l[k]
			new_pattern = sum/count
			pattern_l[l] = new_pattern
		look_table = []
		for k in range(block_l.shape[0]):
			block = block_l[k]
			min = float('inf')
			selected_l = 0
			for l in range(level):
				pattern = pattern_l[l]
				distance = np.linalg.norm(block - pattern)
				if distance <= min:
					selected_l = l
					min = distance
			look_table.append(selected_l)

		sum = 0
		for l in range(level):
			for k in range(block_l.shape[0]):
				if look_table[k] == l:
					sum = sum + np.linalg.norm(block_l[k]-pattern_l[l])
		distortion1 = sum / block_l.shape[0]
		print(distortion1)
		d_l.append(distortion1)
	np.save('pattern10_256.npy',pattern_l)

def VQ_M_test(img,level):
	img = np.asarray(img)
	H = img.shape[0]
	W = img.shape[1]
	count = 0
	block_l = []
	pattern_l = []
	for i in range(0,H,4):
		for j in range(0,W,4):
			block = get_window(img,i,j)
			block_l.append(block)
	block_l = np.asarray(block_l)
	look_table = []
	pattern_l = np.load('pattern10_256.npy')
	for k in range(block_l.shape[0]):
		block = block_l[k]
		min = float('inf')
		selected_l = 0
		for l in range(level):
			pattern = pattern_l[l]
			distance = np.linalg.norm(block - pattern)
			if distance <= min:
				selected_l = l
				min = distance
		look_table.append(selected_l)
	k = 0
	img = np.asarray(img)
	for i in range(0,H,4):
		for j in range(0,W,4):
			img[i:i+4,j:j+4] = pattern_l[look_table[k]]
			k = k + 1
	img = Image.fromarray(img.astype('uint8'), 'L')
	return img

#VQ_M_train(img_l,256)
img = VQ_M_test(img1,256)
#img = VQ(img1,256)
imggt_np = np.asarray(img1)
img_np = np.asarray(img)
psnr = PSNR(imggt_np,img_np)
print(psnr)
img1.save("img1gt.jpg")
img.save("img1256.png")
