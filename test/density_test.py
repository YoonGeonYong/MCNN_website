import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import Crowded

model = Crowded()
model.load_model('./mcnn/trained_B.pth')

for i in range(20):
    num = np.random.randint(1, 100)
    img_path = '/Users/leejuchan/workspace/projects/CrowdCounting/MCNN_svishwa/data/original/shanghaitech/part_B_final/test_data/images/IMG_'+str(num)+'.jpg'

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    dm = model.density_map(img)
    x, y = model.density_point(dm)

    '''
        밀집도가 높을 수록 커널크기에 영향을 많이 받음,
        커널 크기에 따른, 인원수 변화율로 밀집도 추정 가능
    '''
    # den = model.density(dm)
    # den = np.array(den)
    # print(den, len(y))
    nums = model.density(dm)
    nums = np.array(nums)
    # nums = (nums - min(nums)) / (max(nums) - min(nums))
    diffs = np.diff(nums)
    diff = abs(round(np.mean(diffs), 4))

    # print(nums, diffs, diff)


    ''' visualizing '''
    _, axe = plt.subplots(1, 3, constrained_layout=True)

    axe[0].set_title('origin')
    axe[0].imshow(img)

    axe[1].set_title('density map')
    axe[1].imshow(dm)

    axe[2].set_title(f'density point (num : {len(y)})')
    axe[2].imshow(img)
    axe[2].scatter(x*4, y*4, color='r', s=5)

    # axe[0].set_title(f'density point (num : {len(y)})')
    # axe[0].imshow(img)
    # axe[0].scatter(x*4, y*4, color='r', s=5)

    # axe[1].set_title(f'differencing (density : {diff})')
    # axe[1].set_ylim(-1200,0)
    # axe[1].plot(diffs)

    plt.show()