'''
    MCNN 모델을 사용해서, crowd counting을 적용하는 클래스
'''
import numpy as np
import cv2
import torch
from mcnn.crowd_count import CrowdCounter


class Crowded():
    def __init__(self):
        self.model = CrowdCounter()
    
    def load_model(self, path):
        # device
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

        # load
        self.model.load_state_dict(torch.load(path))
        self.model.to(device)
        self.model.eval()
    
    def density_map(self, img):
        # img 처리
        _in = img.reshape(1, 1, img.shape[0], img.shape[1]) # (1, 1, r, c)

        # model
        _out = self.model(_in)
        dm = _out.detach().cpu().numpy()
        dm = dm.squeeze() # (r, c)

        return dm
    
    # local maxima (maximum & median filter)
    def density_point(self, dm):
        maxmask = cv2.dilate(dm, np.ones((3,3)), iterations=4)
        medmask = cv2.medianBlur(dm, ksize=3)

        maxima = (dm == maxmask)
        med = (dm >= medmask + 0.025)
        result = maxima & med
        y, x = np.nonzero(result)

        return x, y
    
    def density(self, dm):
        nums = []

        for i in range(2, 7):
            kernel = np.ones((i, i))
            maxmask = cv2.dilate(dm, kernel, iterations=3)
            medmask = cv2.medianBlur(dm, ksize=3)

            maxima = (dm == maxmask)
            med = (dm >= medmask + 0.025)
            result = maxima & med
            y, x = np.nonzero(result)

            nums.append(len(y))
        diffs = np.diff(nums)
        den = round(abs(np.mean(diffs)), 2)

        return den

# app에 모델 추가
def init_model(app, path):
    model = Crowded()
    model.load_model(path)

    print('load model...')
    app.config['model'] = model

def get_model(app):
    return app.config.get('model')
