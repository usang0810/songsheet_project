import cv2
import os
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

# 모폴로지연산 - 열기
def opening(src, k):
    opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, k)

    return opening

# 모폴로지연산 - 닫기
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
        else:
            cv2.rectangle(dst, (x, y), (x+width, y+height), 0, -1)

    return dst, label_ary

# 2차 레이블링 - 1차 레이블링을 통해 나눈 영역들을에서 요소들을 레이블링
def second_labeling(label_ary, ary_inv):

    for i in range(len(label_ary)):
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(ary_inv[i])

        for j in range(1, int(cnt)):
            x, y, width, height, area = stats[j] # stats는 1부터 시작
            cv2.rectangle(label_ary[i], (x, y), (x+width, y+height), 0, 1)
        
        cv2.imshow("roi"+str(i), label_ary[i])

def second_labeling(src):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    # dst = np.zeros(src.shape, dtype = src.dtype)
    label_ary = []
    for i in range(1, int(cnt)):
        
        x, y, width, height, area = stats[i] # stats는 1부터 시작 0은 이미지 전체영역

        label_ary.append(stats[i])
        cv2.rectangle(src, (x, y), (x+width, y+height), 255, 1)

    return src, label_ary

def make_roi(src, ary):
    roi_img = []
    
    for i in ary:
        img = src[i[1]:i[1]+i[3], i[0]:i[0]+i[2]] # 0:x, 1:y, 2:width, 3:height, 4:area
        roi_img.append(img)
        
    return roi_img

# 템플릿 매칭
def templating(src, temp):

    w, h = temp.shape[::-1]
    res = cv2.matchTemplate(src, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95 # 정확도
    loc = np.where(res >= threshold)

    ary = [] # 좌표값들
    for pt in zip(*loc[::-1]):
        ary2 = [pt]
        bottom_right = pt[0] + w, pt[1] + h
        ary2.append(bottom_right)
        # ary.append([(pt[0]*2+w)/2, (pt[1]*2+h)/2])
        ary.append(ary2)
        
        cv2.rectangle(src, pt, bottom_right, 0, 1)
        # cv2.circle(src, (int((pt[0]*2+w)/2), int((pt[1]*2+h)/2)), 5, 0, 2)

    return src, ary

# 템플릿 디렉토리에서 파일 가져오기
def search(dirname):
    filenames = os.listdir(dirname)
    ary = []

    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ary.append(full_filename)
        print (full_filename)

    return ary

dirname = "./template"
template_names = search(dirname)
src = "./images/bears.jpg"
src = imageLoad(src)
src_not = cv2.bitwise_not(src)
src_not = binaryTo(src_not) # 이진화를 하지않으면 레이블링의 오차범위가 넓어짐

sharp = sharpTo(src)
binary = binaryTo(sharp)
line, fiveline = Findfiveline(binary) # 오선의 좌표값 추출

del_line = delete_line(src, line) # 오선삭제
del_line = binaryTo(del_line) # 이진화작업

# 모폴로지 연산을 위한 커널 사각형(2 x 2) 생성, 악보크기에 따라 커널 사각형의 값이 적절해야함
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

mophol_img = opening(del_line, kernal) # 모폴로지 연산을 이용해 오선을 제거한 부분을 자연스럽게 매꿈
first_labeling, label_ary = first_labeling(src_not) # 1차 레이블링 레이블한 범위를 배열에 저장
dst = cv2.bitwise_and(mophol_img, mophol_img, mask = first_labeling) # and연산을 이용해 mophol_img에서 mask부분만 나타냄

# cv2.imshow("first", first_labeling)

first_inv = cv2.bitwise_not(first_labeling) # 배경이미지에 관심영역을 넣기위한 labeling이미지의 inv
add = cv2.add(dst, first_inv) # 배경과 잘라낸 이미지 합성
add_inv = cv2.bitwise_not(add)

second_labeling2, label_ary2 = second_labeling(add_inv)
second_labeling2 = cv2.bitwise_not(second_labeling2)

# temp = "./template/high.jpg"
# temp = imageLoad(temp)
# temp = binaryTo(temp)
# template, high = templating(add, temp)

# temp2 = "./template/note_8.jpg"
# temp2 = imageLoad(temp2)
# temp2 = binaryTo(temp2)
# template2, high = templating(add, temp2)

# temp3 = "./template/note_4.jpg"
# temp3 = imageLoad(temp3)
# temp3 = binaryTo(temp3)
# template3, high = templating(add, temp3)

# temp4 = "./template/note_2.jpg"
# temp4 = imageLoad(temp4)
# temp4 = binaryTo(temp4)
# template4, high = templating(add, temp4)

i = 1
for template_name in template_names:
    temp = imageLoad(template_name)
    temp = binaryTo(temp)
    ary = []
    cv2.imshow("temp"+str(i), temp)
    add, ary = templating(add, temp)
    print("temp"+str(i),":",ary)

    i += 1

cv2.imshow("add", add)

# cv2.imshow("second_labeling2", second_labeling2)
# cv2.imshow("temp", template)
# cv2.imshow("second_label", second_labeling2)
# print(label_ary2)
# 모폴로지 연산을 이용해 머리좌표만 남김
# mophol_img2 = opening(add, kernal)
# mophol_img2 = closing(add, kernal)
# mophol_img2 = closing(mophol_img2, kernal)
# mophol_img2 = closing(mophol_img2, kernal)
# mophol_img2 = opening(mophol_img2, kernal)
# mophol_img2 = cv2.dilate(add, kernal, iterations=1)
# mophol_img2 = cv2.erode(mophol_img2, kernal, iterations=1)
# cv2.imshow("mophol_img2", mophol_img2)

# label_ary_inv = [] # 2차레이블링을 위한 레이블링의 inv을 저장하기 위한 배열
roi_img2 = make_roi(dst, label_ary2) # 레이블 배열을 이용해 영역마다 나눔
# template, high = templating(roi_img2, temp)
# print(high)

for i in range(len(roi_img2)):
    # cv2.imshow("roi"+str(i), roi_img2[i])
    inv = cv2.bitwise_not(roi_img2[i]) # 흑백변환
    # label_ary_inv.append(inv)
    # cv2.imshow("roi_inv"+str(i), label_ary_inv[i])

# second_labeling(roi_img, label_ary_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()