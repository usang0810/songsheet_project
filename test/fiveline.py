import cv2
import numpy as np
from matplotlib import pyplot as plt

# imageload method
def imageLoad(src):
    image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    return image

# 이미지 외곽선 뚜렷하게
def sharpTo(src):
    kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    output = cv2.filter2D(src, -1, kernel_sharpen_1)
    return output

# 이미지 이진화
def binaryTo(src):
    # 단순 이진화
    ret, output = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)

    return output

# 오선의 좌표값 찾기(이미지의 행에 있는 검은색 화소가 70%이상을 차지하고 있다면 오선으로 취급, 여기서 70%의 값은 80으로 기준)
def Findfiveline(src):
    rows, cols = src.shape
    print('rows = ', rows, 'cols = ', cols)
    
    fiveline = []
    all_line = []
    line = []
    count = 0
    pre = 0 # 이전값

    for i in range(0, rows):
        avg = sum(src[i], 0) / rows
        if avg < 80:
            all_line.append(i)
            if i < pre+3:
                continue
            else:
                pre = i
                line.append(i)
                count = count + 1
            
            if count >= 5:
                fiveline.append(line)
                count = 0
                line = []

    return all_line, fiveline

def delete_line(src, fiveline):
    for i in range(len(fiveline)):
        src[fiveline[i]] = 255

    return src


src = "./images/bears.jpg"
src = imageLoad(src)

sharp = sharpTo(src)
binary = binaryTo(sharp)
line, fiveline = Findfiveline(binary) # 오선의 좌표값 추출
print(line, fiveline)

del_line = delete_line(src, line)
del_line = binaryTo(del_line)

print(cv2.__version__)

cv2.imshow("del_line", del_line)
cv2.waitKey(0)
cv2.destroyAllWindows()