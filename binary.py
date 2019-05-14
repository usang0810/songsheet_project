import cv2

original = cv2.imread("./images/moonlight.png", cv2.IMREAD_GRAYSCALE)
print(original.shape)
# resized = cv2.resize(original, (80, 50), interpolation=cv2.INTER_CUBIC)
# print(resized.shape)
cuttingimg = original[400:450, 300:380]

ret, binary_img = cv2.threshold(cuttingimg, 150, 1, cv2.THRESH_BINARY)


for i in range(0, 50):
    for j in range(0, 80):
        print(binary_img[i][j], end = ' ')
    print()

cv2.imshow('img', cuttingimg)
cv2.waitKey(0)
cv2.destroyAllWindows()