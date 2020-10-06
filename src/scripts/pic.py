import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_image(fName):
    img = Image.open(fName)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


# blueIsh = [82,  43, 227]
# imgDir = '/home/mina/Dropbox/APRIL-MINA/EXP3_Generation/videos/VAE_42-200/im_seq_grid_N2S/grid_origin_long1/dec_long1_r0.5_l2'
# imgSelection = sorted(os.listdir(imgDir))
#
# n_imgs = len(imgSelection)
#
# imgs = []
# for f in imgSelection:
#     img = load_image(os.path.join(imgDir, f))
#     mask = np.logical_and((75 < img[:, :, 0]), (img[:, :, 0] < 85))
#     img[mask] = 0
#
#     imgs.append(img)
#
# print(len(imgs))
# imgs = np.stack(imgs)
# imgSum = imgs.max(axis=0)
# imgSum = imgSum / imgSum.max()
# print(mask.shape)
# plt.imshow(mask * 255)
#
# # loop
# step = 280
#
# # Create the end image (1st define shape)
# endImgS = list(imgs[0].shape)
# endImgS[1] = endImgS[1] + step * n_imgs
# print(imgs.shape)
# print(endImgS)
# endImg = np.zeros(endImgS)
#
# # Fill the end image
# for i in range(n_imgs):
#     subImg = np.maximum(imgs[i], endImg[:, i * step: i * step + 1280])
#     endImg[:, i * step: i * step + 1280] = subImg
#
# # Plot
# endImg = endImg[50:600, 410:3380, :]
# plt.imshow(endImg/255.)
# plt.axis('off')
# # plt.show()
# plt.savefig(os.path.join(imgDir, "dec_long1_r0.5_l2.eps"), format='eps', dpi=1000, bbox_inches="tight", pad_inches=0)

imgDir = '/home/mina/Dropbox/APRIL-MINA/EXP3_Generation/videos/VAE_42-200/im_seq_grid_N2S/grid_origin_long1/img'


fig, ax = plt.subplots(4, 2, figsize=(30, 18), dpi=100000,  gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)

imgSelection = ['dec_long1_r0.5_l2.eps',
                'dec_long1_r3_l2.eps',
                'dec_long1_r6_l2.eps',
                'dec_long1_r3_l1.eps',
                'dec_long1_r10_l2.eps',
                'rev_l2_long1_r3.eps',
                'dec_long2_r6_l2.eps',
                'dec_long1_r3_l3.eps'
                ]

img0 = load_image(os.path.join(imgDir, imgSelection[0]))
img1 = load_image(os.path.join(imgDir, imgSelection[1]))
img2 = load_image(os.path.join(imgDir, imgSelection[2]))
img3 = load_image(os.path.join(imgDir, imgSelection[3]))

img4 = load_image(os.path.join(imgDir, imgSelection[4]))
img5 = load_image(os.path.join(imgDir, imgSelection[5]))
img6 = load_image(os.path.join(imgDir, imgSelection[6]))
img7 = load_image(os.path.join(imgDir, imgSelection[7]))

plt.subplot(421)
plt.imshow(img0, interpolation='none')
plt.axis('off')
# plt.title('img0')


plt.subplot(422)
plt.imshow(img4)
plt.axis('off')
# plt.title('img4')

plt.subplot(423)
plt.imshow(img1)
plt.axis('off')
# plt.title('img1')

plt.subplot(424)
plt.imshow(img5)
plt.axis('off')
# plt.title('img5')

plt.subplot(425)
plt.imshow(img2)
plt.axis('off')
# plt.title('img2')

plt.subplot(426)
plt.imshow(img6)
plt.axis('off')
# plt.title('img6')

plt.subplot(427)
plt.imshow(img3)
plt.axis('off')
# plt.title('img3')

plt.subplot(428)
plt.imshow(img7)
plt.axis('off')
# plt.title('img7')


plt.show()