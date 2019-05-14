import cv2
import numpy as np

image = cv2.imread("./images/Moonlight Shadow Flauta-1.png", cv2.IMREAD_GRAYSCALE)
w, h = image.shape[::-1]
image = cv2.resize(image, (int(w * 0.75), int(h * 0.75)))
print(w, h)
# cv2.resize(image, )
quarter = cv2.imread("./template/quarter.png", cv2.IMREAD_GRAYSCALE)
th, tw = quarter.shape[:2]
print(th, tw)
quarter = cv2.resize(quarter, (int(tw * 0.3), int(th * 0.3)))

cv2.imshow('quarter', quarter)

image_draw = image.copy()
res = cv2.matchTemplate(image, quarter, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val, max_val, min_loc, max_loc)

top_left = max_loc
match_val = max_val
bottom_right = (top_left[0] + tw, top_left[1] + th)
cv2.rectangle(image_draw, top_left, bottom_right, 0, 2)

cv2.imshow('aa', image_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()