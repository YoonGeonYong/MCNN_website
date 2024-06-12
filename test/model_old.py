# from mcnn.crowd_count import CrowdCounter
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

from model import MCNN

 
# model
model = MCNN(3e-4)
model.load_state_dict(torch.load('./mcnn_trained.pth'))

# img
img = cv2.imread('../upload/input.jpg', 0)    # (r, c)
_in = img.reshape(1, img.shape[0], img.shape[1]) # (1, 1, r, c)
_in = torch.Tensor(_in)

_out = model(_in)
dm = _out.detach().cpu().numpy() # (1, 1, r, c)
dm = dm.squeeze()                # (r, c)


''' visualizing '''
_, axe = plt.subplots(1, 3, constrained_layout=True)

axe[0].imshow(img)
axe[1].imshow(dm)
plt.show()