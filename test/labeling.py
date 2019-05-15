import cv2
import numpy as np
from matplotlib import pyplot as plt

# imageload method
def imageLoad(src):
    image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    return image

# 이미지 이진화
def binaryTo(src):
    # 단순 이진화
    ret, output = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY) # INV 붙이면 흑백
    # 흑백으로 하는이유는 레이블링을 할때 흰색값을 인식하기 때문

    return output

src = "./images/bears.jpg"
src = imageLoad(src)
src2 = cv2.bitwise_not(src)
gray = src2.copy()
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
binary = binaryTo(gray)

# ret, labels = cv2.connectedComponents(binary)
# print('ret = ', ret)

# dst = np.zeros(src.shape, dtype = src.dtype)
# for i in range(1, ret):
#     r = np.random.randint(256)
#     g = np.random.randint(256)
#     b = np.random.randint(256)
#     dst[labels == i] = [b, g, r]
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
# print(cnt)
# print(labels)
# print(stats)
# print(centroids)

dst = np.zeros(src.shape, dtype = src.dtype)
# for i in range(1, int(cnt)):
#     r = np.random.randint(256)
#     g = np.random.randint(256)
#     b = np.random.randint(256)
#     dst[labels == i] = [b, g, r]

label_ary = []
for i in range(1, int(cnt)):
    
    x, y, width, height, area = stats[i] # stats는 1부터 시작
    print(i, area)
    if area >3500:
        label_ary.append(stats[i])
        cv2.rectangle(dst, (x, y), (x+width, y+height), 255, -1)
    # else:
    #     cv2.rectangle(dst, (x, y), (x+width, y+height), 0, -1)

# for stat in stats:
#     cv2.rectangle(src, (stat[0], stat[1]), (stat[0]+stat[2], stat[1]+stat[3]), 0, 1)
#     print(stat)
cv2.imshow("dst", dst)
cv2.imshow("src", src)
dst2 = cv2.bitwise_and(src, src, mask = dst)
print(label_ary)
cv2.imshow('res', binary)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()