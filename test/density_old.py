import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# from mcnn.model import Crowded

num = np.random.randint(1, 100)
img_path = '/Users/leejuchan/workspace/projects/CrowdCounting/MCNN_svishwa/data/original/shanghaitech/part_B_final/test_data/images/IMG_'+str(num)+'.jpg'
# img_path = '/Users/leejuchan/Downloads/test.jpeg'

# model = Crowded()
# model.load_model('./mcnn/trained_B.pth')
##
import torch
from model import MCNN
model = MCNN(3e-4)
model.load_state_dict(torch.load('./mcnn_trained.pth'))

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

_in = img.reshape(1, img.shape[0], img.shape[1]) # (1, 1, r, c)
_in = torch.Tensor(_in)
_out = model(_in)
dm = _out.detach().cpu().numpy() # (1, 1, r, c)
dm = dm.squeeze()

maxima = cv2.dilate(dm, None, iterations=3)
med = cv2.medianBlur(dm, ksize=3)

maxmask = (dm == maxima)
medmask = (dm >= med + 0.02)
mask = maxmask & medmask

y, x = np.nonzero(mask)
##
# dm = model.density_map(img)
# x, y = model.density_point(dm)

'''
    밀집도가 높을 수록 커널크기에 영향을 많이 받음,
    커널 크기에 따른, 인원수 변화율로 밀집도 추정 가능
'''
# den = model.density(dm)
# den = np.array(den)
# den = (den - max(den)) / (max(den) - min(den))
# dden = np.diff(den) / 0.1
# print(max(abs(dden)), len(y))
# print(den, len(y))


''' visualizing '''
_, axe = plt.subplots(1, 3, constrained_layout=True)

axe[0].set_title('origin')
axe[0].imshow(img)

axe[1].set_title('density map')
axe[1].imshow(dm)

axe[2].set_title(f'density point (num : {len(y)})')
axe[2].imshow(img)
axe[2].scatter(x*4, y*4, color='r', s=5)

plt.show()