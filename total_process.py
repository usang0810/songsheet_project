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

# labeling을 할때는 흑백전환 후 해야함
# 1차 레이블링 - 오선주의 영역을 레이블링
def first_labeling(src): 
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src)

    dst = np.zeros(src.shape, dtype = src.dtype)
    label_ary = []
    for i in range(1, int(cnt)):
        
        x, y, width, height, area = stats[i] # stats는 1부터 시작
        # print(i, area)
        if area >3500: # label의 넓이가 3500 이상일때만 사각형
            label_ary.append(stats[i])
            cv2.rectangle(dst, (x, y), (x+width, y+height), 255, -1)

    return dst, label_ary

# 2차 레이블링 - 1차 레이블링을 통해 나눈 영역들을에서 요소들을 레이블링
def second_labeling(label_ary, ary_inv):

    for i in range(len(label_ary)):
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(ary_inv[i])

        for j in range(1, int(cnt)):
            x, y, width, height, area = stats[j] # stats는 1부터 시작
            cv2.rectangle(label_ary[i], (x, y), (x+width, y+height), 0, 1)
        
        cv2.imshow("roi"+str(i), label_ary[i])

def make_roi(src, ary):
    roi_img = []
    
    for i in ary:
        img = src[i[1]:i[1]+i[3], i[0]:i[0]+i[2]] # 0:x, 1:y, 2:width, 3:height, 4:area
        roi_img.append(img)
        
    return roi_img

src = "./images/bears.jpg"
src = imageLoad(src)
src_not = cv2.bitwise_not(src)
src_not = binaryTo(src_not) # 이진화를 하지않으면 레이블링의 오차범위가 넓어짐

sharp = sharpTo(src)
binary = binaryTo(sharp)
line, fiveline = Findfiveline(binary) # 오선의 좌표값 추출

del_line = delete_line(src, line) # 오선삭제
del_line = binaryTo(del_line) # 이진화작업

kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # 모폴로지 연산을 위한 커널 사각형(5 x 5) 생성
mophol_img = opening(del_line, kernal) # 모폴로지 연산을 이용해 오선을 제거한 부분을 자연스럽게 매꿈
first_labeling, label_ary = first_labeling(src_not) # 1차 레이블링 레이블한 범위를 배열에 저장
dst = cv2.bitwise_and(mophol_img, mophol_img, mask = first_labeling) # and연산을 이용해 mophol_img에서 mask부분만 나타냄

print(label_ary)
label_ary_inv = [] # 2차레이블링을 위한 레이블링의 inv을 저장하기 위한 배열
roi_img = make_roi(dst, label_ary) # 레이블 배열을 이용해 영역마다 나눔
for i in range(len(roi_img)):
    # cv2.imshow("roi"+str(i), roi_img[i])
    inv = cv2.bitwise_not(roi_img[i]) # 흑백변환
    label_ary_inv.append(inv)
    # cv2.imshow("roi_inv"+str(i), label_ary_inv[i])

second_labeling(roi_img, label_ary_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()