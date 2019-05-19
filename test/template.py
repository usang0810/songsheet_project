import cv2
import numpy as np

image = cv2.imread("./images/bears.jpg", cv2.IMREAD_GRAYSCALE)
quarter = cv2.imread("./template/quarter.png", cv2.IMREAD_GRAYSCALE)
half_space = cv2.imread("./template/half-note-space.png", cv2.IMREAD_GRAYSCALE)
half_line = cv2.imread("./template/half-not-line.png", cv2.IMREAD_GRAYSCALE)

dst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# #1
# R1 = cv2.matchTemplate(image, quarter, cv2.TM_SQDIFF_NORMED)
# minval, _, minLoc, _ = cv2.minMaxLoc(R1)
# print('tm_sqdiff_mormed : ', minval, minLoc)
# w, h = quarter.shape[:2]
# print('w = ', w, 'h = ', h)
# cv2.rectangle(dst, minLoc, (minLoc[0] + h, minLoc[1] + w), (255, 0, 0), 2)

# #2
# R2 = cv2.matchTemplate(image, half_space, cv2.TM_CCORR_NORMED)
# _, maxval, _, maxLoc = cv2.minMaxLoc(R2)
# print('tm_ccorr_normed : ', maxval, maxLoc)
# w, h = half_space.shape[:2]
# cv2.rectangle(dst, maxLoc, (maxLoc[0] + h, maxLoc[1] + w), (0, 255, 0), 2)

#3
w, h = quarter.shape[::-1]
print('w = ', w, 'h = ', h)
R3 = cv2.matchTemplate(image, quarter, cv2.TM_CCOEFF_NORMED)
minval, maxval, minloc, maxLoc = cv2.minMaxLoc(R3)
print('tm_ccoeff_normed : ', maxval, maxLoc)

cv2.rectangle(dst, maxLoc, (maxLoc[0] + h, maxLoc[1] + w), (0, 0, 255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()