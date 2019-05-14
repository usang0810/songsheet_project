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
    
    # 흑백반전 후 이진화
    binary = cv2.bitwise_not(output)
    bw = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    return output, bw

# 오선 추출
def Extraction(src):
    horizontal = np.copy(src)
    vertical = np.copy(src)

    # horizontal method (오선 추출)
    cols = horizontal.shape[1]
    horizontal_size = cols / 5 # 값이 커질수록 오차가 넓어짐

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontal_size), 1)) # 형변환안하면 오류
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    horizontal = cv2.bitwise_not(horizontal) # 흑백반전

    # vertical method (오선을 제외한 나머지 추출)
    rows = vertical.shape[0]
    vertical_size = rows / 250 # 값 조정범위가 애매함

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_size)))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical = cv2.bitwise_not(vertical) # 흑백반전

    return horizontal, vertical

# vertical 마무리
def smoothTo(src):
    edges = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    dilate = cv2.dilate(edges, kernel)
    # Step 3
    smooth = np.copy(src)
    # Step 4
    smooth = cv2.blur(smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(edges != 0)
    src[rows, cols] = smooth[rows, cols]

    return src

# 오선의 좌표값 찾기(이미지의 행에 있는 검은색 화소가 70%이상을 차지하고 있다면 오선으로 취급, 여기서 70%의 값은 80으로 기준)
# 선의 두께에 대한 오차도 설정해줘야 함(선의 두께를 3으로 기준)
def Findfiveline(src):
    rows, cols = src.shape
    fiveline = []
    line = []
    count = 0
    pre = 0 # 이전값

    for i in range(0, rows):
        avg = sum(src[i], 0) / rows
        if avg < 80:
            print(i, avg)
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

    return fiveline

# def garbage_del(src, fiveline): # 필요없을듯

#     avg = []
#     avg.append(0)
#     roi = []
#     roi

#     for i in range(len(fiveline)):
#         if fiveline[-1] == fiveline[i]:
#             break
#         # print(fiveline[i][4], fiveline[i+1][0])
#         avg.append(round((fiveline[i][4] + fiveline[i+1][0]) / 2))

#     avg[0] = fiveline[0][0] - (avg[1] - fiveline[0][4])
#     rows, cols = src.shape
#     if fiveline[-1][4] + (avg[1] - fiveline[0][4]) > rows:
#         avg.append(rows)
#     else:
#         avg.append((fiveline[-1][4] + (avg[1] - fiveline[0][4])))

#     # print(avg)
#     for i in range(len(avg)):
#         if avg[-1] == avg[i]:
#             break
#         roi.append(src[avg[i]:avg[i+1], 0:cols])
#         cv2.imshow('roi'+str(i+1), roi[i])re(res >= threshold)

# 오선을 관심영역으로 추출
def Roifiveline(src, fiveline):
    rows, cols = src.shape
    roi = []
    for i in range(len(fiveline)):
        print(fiveline[i])
        roi.append(src[fiveline[i][0]:fiveline[i][4], 0:cols])
        cv2.imshow('roi'+str(i), roi[i])

    return roi, fiveline

# 새로운 이미지 생성 후 오선 그리기
def createSheet(src):
    rows, cols = src.shape
    image = np.zeros((rows, cols, 1), dtype="uint8")
    image[:] = 255

    return image

def Addfiveline(src, fiveline):
    for i in range(len(fiveline)):
        for j in range(len(fiveline[i])):
            src[fiveline[i][j]] = 0

    return src

def defiveline(src, fiveline):
    for i in range(len(fiveline)):
        for j in range(len(fiveline[i])):
            src[fiveline[i][j]] = 255

    return src


src = "./images/moonlight.png"

src = imageLoad(src)
sharp = sharpTo(src)
binary, binary2 = binaryTo(sharp)
horizon, verti = Extraction(binary2)
verti = smoothTo(verti)

# cv2.imshow('img', src)
# cv2.imshow('sharp', sharp)
# cv2.imshow('binary', binary)
# cv2.imshow('binary2', binary2)
# cv2.imshow("horizontal", horizon) # 오선 좌표만 있는이미지
cv2.imshow("vertical", verti)

# cv2.imshow("final", verti)
# print(verti[0])

fiveline = Findfiveline(binary) # 오선의 좌표값 추출
de_src = defiveline(src, fiveline)
roi = Roifiveline(binary, fiveline) # 오선을 기준으로 한 관심영역설정
new_sheet = createSheet(binary) # 새로운 이미지 생성
fiveline_sheet = Addfiveline(new_sheet, fiveline) # 새로운 이미지에 오선 그리기

cv2.imshow('img2', de_src)
cv2.imshow('image', fiveline_sheet)
cv2.waitKey(0)
cv2.destroyAllWindows()