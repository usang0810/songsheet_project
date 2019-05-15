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
    ret, output = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)

    return output


src = "./images/bears.jpg"
src = imageLoad(src)

gx_k = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
gy_k = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

edge_gx = cv2.filter2D(src, -1, gx_k)
edge_gy = cv2.filter2D(src, -1, gy_k)

sobelx = cv2.Sobel(src, -1, 1, 0, ksize = 3)
sobely = cv2.Sobel(src, -1, 0, 1, ksize = 3)

merged1 = np.hstack((src, edge_gx, edge_gy, edge_gx + edge_gy))
merged2 = np.hstack((src, sobelx, sobely, sobelx + sobely))
merged = np.vstack((merged1, merged2))

cv2.imshow("sobely", sobelx + sobely)
# cv2.imshow("sobel", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()