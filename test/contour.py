import cv2
import numpy as np

image = cv2.imread("./images/bears.jpg", cv2.IMREAD_GRAYSCALE)
image2 = image.copy()

ret, imthres = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

contour, hierarchy = cv2.findContours(imthres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contour, -1, 0, 4)

print(contour)

for i in contour:
    for j in i:
        cv2.circle(image, tuple(j[0]), 1, 0, -1)

cv2.imshow("con", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
