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

src2 = src.copy()
h, w = src.shape[:2]

edges = cv2.Canny(src, 100, 200)
sobelx = cv2.Sobel(src, -1, 1, 0, ksize = 1)
sobely = cv2.Sobel(src, -1, 0, 1, ksize = 1)
cv2.imshow("edges", edges)
cv2.imshow("sobelx", sobelx)
cv2.imshow("sobely", sobely)

lines = cv2.HoughLines(sobely, 1, np.pi/180, 500)
fiveline = []
for line in lines:
    print(line)
    fiveline.append(line[0][0])
    r, theta = line[0]
    tx, ty = np.cos(theta), np.sin(theta)
    x0, y0 = tx*r, ty*r
    cv2.circle(src2, (abs(x0), abs(y0)), 3, 0, -1)

    x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
    x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)

    cv2.line(src2, (x1, y1), (x2, y2), 0, 1)

for i in range(len(fiveline)):
    src[int(fiveline[i])] = 255

print(fiveline)
merged = np.hstack((src, src2))
cv2.imshow("merged", merged)
# cv2.imshow("canny", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()