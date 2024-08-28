import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import Crowded

model = Crowded()
model.load_model('./mcnn/trained_B.pth')

# 이미지 로드
def load_img(part, type, num):
    img_path = '/Users/leejuchan/workspace/projects/CrowdCounting/MCNN_svishwa/data/original/shanghaitech/part_' +part+ '_final/' +type+ '_data/images/IMG_' +str(num)+ '.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return img

# k에 따른 density point
def plot_density_point(num):
    img = load_img('A', 'test', num)
    dm = model.density_map(img)
    den = model.density(dm)

    _, axe = plt.subplots(3, 5, constrained_layout=True)

    for i in range(2, 7):
        maxmask = cv2.dilate(dm, np.ones((i, i)), iterations=4)
        medmask = cv2.medianBlur(dm, ksize=3)

        maxima = (dm == maxmask)
        med = (dm >= medmask + 0.025)
        y, x = np.nonzero(maxima & med)

        # plot
        axe[0,i-2].set_title(f'k ={i} (num: {len(x)})')
        axe[0,i-2].set_axis_off()
        axe[0,i-2].imshow(dm)

        axe[1,i-2].set_axis_off()
        axe[1,i-2].imshow(maxmask)

        axe[2,i-2].set_axis_off()
        axe[2,i-2].imshow(dm)
        axe[2,i-2].scatter(x, y, color='r', s=5)
    
    plt.suptitle(f'density: {den}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()

    

    

# img, density map, density point 시각화
def plot_density(num):
    img = load_img('B', 'test', num)
    dm = model.density_map(img)
    x, y = model.density_point(dm)
    den = model.density(dm)         # 밀집도가 높을 수록 커널크기에 영향을 많이 받음, 커널 크기에 따른, 인원수 변화율로 밀집도 추정 가능

    # 시각화
    _, axe = plt.subplots(1, 3, constrained_layout=True)
    axe[0].set_title('origin')
    axe[0].set_axis_off()
    axe[0].imshow(img)

    axe[1].set_title('density map')
    axe[1].set_axis_off()
    axe[1].imshow(dm)

    axe[2].set_title(f'density point (num : {len(y)})')
    axe[2].set_axis_off()
    axe[2].imshow(img)
    axe[2].scatter(x*4, y*4, color='r', s=5)
    plt.show()


''' main '''
for i in range(1, 182): # 182 316
    plot_density_point(i)

# plot_density(32)

# for i in range(20):
#     n = np.random.randint(1, 100)
#     plot_density(n)