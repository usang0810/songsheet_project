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

def opening(src, k):
    opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, k)

    return opening

def closing(src, k):
    closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, k)

    return closing

# 템플릿 매칭
def templating(src, temp):

    w, h = temp.shape[::-1]
    res = cv2.matchTemplate(src, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.70 # 정확도
    loc = np.where(res >= threshold)

    ary = [] # 좌표값들
    for pt in zip(*loc[::-1]):
        bottom_right = pt[0] + w, pt[1] + h
        ary.append([(pt[0]*2+w)/2, (pt[1]*2+h)/2])
        
        cv2.rectangle(src, pt, bottom_right, 0, 2)
        # cv2.circle(src, (int((pt[0]*2+w)/2), int((pt[1]*2+h)/2)), 5, 0, 2)

    return src, ary

# 계이름 추출
# def Findsyllable(fiveline, notes):

#     for k in range(len(notes)):
#         for i in range(len(fiveline)):
#             for j in range(len(fiveline[i])):
            

src = "./images/bears.jpg"
quarter = "./template/quarter.png"
src = imageLoad(src)
quarter = imageLoad(quarter)

w, h = quarter.shape[::-1]
quarter = cv2.resize(quarter, (int(w * 0.35), int(h * 0.35))) # 템플릿이미지 사이즈 조절

sharp = sharpTo(src)
binary = binaryTo(sharp)
line, fiveline = Findfiveline(binary) # 오선의 좌표값 추출
print(line, fiveline)
del_line = delete_line(src, line)
del_line = binaryTo(del_line)

kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # 모폴로지 연산을 위한 커널 사각형(5 x 5) 생성
# 침식 - 팽창 3회 - 침식
mophol_img = opening(del_line, kernal)
# mophol_img = closing(mophol_img, kernal)
# mophol_img = closing(mophol_img, kernal)
# mophol_img = closing(mophol_img, kernal)
# mophol_img = opening(mophol_img, kernal)

# garbage_del(binary, fiveline)

cv2.imshow('image', binary)
cv2.imshow('temp', quarter)
# cv2.waitKey()
# cv2.imshow('del_line', del_line)
# cv2.waitKey()
cv2.imshow('mophol', mophol_img)
# cv2.rectangle(mophol_img, (0, 0), (50, 50), 0, 2)
mophol_img, notes = templating(mophol_img, quarter)
mophol_img2, notes2 = templating(binary, quarter)
cv2.imshow('mophol', mophol_img)
cv2.imshow('mophol2', mophol_img2)
print(len(notes), notes)
# print(len(notes2), notes2)
cv2.waitKey()
cv2.destroyAllWindows()