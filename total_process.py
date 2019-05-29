import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

class Note:
    def __init__(self):
        self.note_head = None
        self.note_beat = None
        self.fline_area = 0
        self.name = None
        self.beat = None
        
    def set_head(self, note_head):
        self.note_head = note_head

    def set_beat(self, note_beat):
        self.note_beat = note_beat

    def set_fline(self, fline_area):
        self.fline_area = fline_area
        
    def get_head(self):
        return self.note_head

    def get_beat(self):
        return self.note_beat

    def get_fline(self):
        return self.fline_area
    
    def __str__(self):
        return '{self.note_head}'

# image load method
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
# def second_labeling(label_ary, ary_inv):

#     for i in range(len(label_ary)):
#         cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(ary_inv[i])

#         for j in range(1, int(cnt)):
#             x, y, width, height, area = stats[j] # stats는 1부터 시작
#             cv2.rectangle(label_ary[i], (x, y), (x+width, y+height), 0, 1)
        
#         cv2.imshow("roi"+str(i), label_ary[i])

def second_labeling(src):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    dst = np.zeros(src.shape, dtype = src.dtype)
    label_ary = []
    for i in range(1, int(cnt)):
        
        x, y, width, height, area = stats[i] # stats는 1부터 시작 0은 이미지 전체영역
        # print(area)
        # label_ary.append(stats[i])
        if 100 <= area <= 200: # label의 넓이가 3500 이상일때만 사각형
            # print("src:",src[y:y+height, x:x+width])
            src_sum = cv2.mean(src[y:y+height, x:x+width]) # 이미지의 평균값이 150이하이면 label_ary에 추가
            # print(src_sum)
            if src_sum[0] < 150:
                label_ary.append(stats[i])
                cv2.rectangle(dst, (x, y), (x+width, y+height), 255, -1)

        else:
            cv2.rectangle(dst, (x, y), (x+width, y+height), 0, -1)
            pass
        # cv2.rectangle(src, (x, y), (x+width, y+height), 255, 1)

    return dst, label_ary

def print_labeling(src, notes):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    dst = np.zeros(src.shape, dtype = src.dtype)
    note_beats = []
    for i in range(1, int(cnt)):
        note_beat = []
        x, y, width, height, area = stats[i] # stats는 1부터 시작 0은 이미지 전체영역
        note_beat.append(x)
        note_beat.append(y)
        note_beat.append(width)
        note_beat.append(height)
        notes[i-1].set_beat(note_beat)

        # print(src[y:y+height, x:x+width])
        '''
        src[y:y+height, x:x+width]
        '''
        cv2.rectangle(dst, (x, y), (x+width, y+height), 255, -1)
        # print(src_sum)
        note_beats.append(note_beat)

    return dst, note_beats, notes

def third_labeling(src):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    label_ary = []
    notes = []
    # print(cnt)
    for i in range(1, int(cnt)):
        x, y, width, height, area = stats[i] # stats는 1부터 시작 0은 이미지 전체영역
        label_ary.append([int((x*2+width)/2), int((y*2+height)/2)])
        # note = Note(note_head = [int((x*2+width)/2), int((y*2+height)/2)])
        note = Note()
        note.set_head([int((x*2+width)/2), int((y*2+height)/2)])

        notes.append(note)
        # print("label_array: ", label_ary[i-1])
        
    return label_ary, notes

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
    # print(res.shape)

    # print("def templating: ", res)

    threshold = 0.90 # 정확도
    loc = np.where(res >= threshold)
    # print("def loc templating: ", loc)
    ary = [] # 좌표값들
    for pt in zip(*loc[::-1]):
        ary2 = [pt]
        bottom_right = pt[0] + w, pt[1] + h
        ary2.append(bottom_right)
        # ary.append([(pt[0]*2+w)/2, (pt[1]*2+h)/2])
        ary.append(ary2)
        
        cv2.rectangle(src, pt, bottom_right, 0, 1)
        # print("top_left", pt)
        # print("bottom_right", bottom_right)
        # cv2.circle(src, (int((pt[0]*2+w)/2), int((pt[1]*2+h)/2)), 5, 0, 2)

    return src, ary

# 템플릿 디렉토리에서 파일 가져오기
def search(dirname):
    filenames = os.listdir(dirname)
    ary = []

    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ary.append(full_filename)
        # print (full_filename)

    return ary

def findnotename(note_heads, fiveline):
    note_names = []
    for i in range(len(note_heads)):
        note_name = []
        for j in range(len(note_heads[i])):
            if note_heads[i][j][1] < fiveline[i][0] - 1:
                note_name.append('PASS')
            elif fiveline[i][0] -1 <= note_heads[i][j][1] <= fiveline[i][0] +1:
                note_name.append('5F')
            elif int((fiveline[i][0] + fiveline[i][1]) / 2) -1 <= note_heads[i][j][1] <= int((fiveline[i][0] + fiveline[i][1]) / 2) +1:
                note_name.append('5E')
            elif fiveline[i][1] -1 <= note_heads[i][j][1] <= fiveline[i][1] +1:
                note_name.append('5D')
            elif int((fiveline[i][1] + fiveline[i][2]) / 2) -1 <= note_heads[i][j][1] <= int((fiveline[i][1] + fiveline[i][2]) / 2) +1:
                note_name.append('5C')
            elif fiveline[i][2] -1 <= note_heads[i][j][1] <= fiveline[i][2] +1:
                note_name.append('4B')
            elif int((fiveline[i][2] + fiveline[i][3]) / 2) -1 <= note_heads[i][j][1] <= int((fiveline[i][2] + fiveline[i][3]) / 2) +1:
                note_name.append('4A')
            elif fiveline[i][3] -1 <= note_heads[i][j][1] <= fiveline[i][3] +1:
                note_name.append('4G')
            elif int((fiveline[i][3] + fiveline[i][4]) / 2) -1 <= note_heads[i][j][1] <= int((fiveline[i][3] + fiveline[i][4]) / 2) +1:
                note_name.append('4F')
            elif fiveline[i][4] -1 <= note_heads[i][j][1] <= fiveline[i][4] +1:
                note_name.append('4E')
            elif fiveline[i][4] +1 < note_heads[i][j][1]:
                note_name.append('PASS')
        note_names.append(note_name)

    return note_names
    

dirname = "./template2"
template_names = search(dirname)
src = "./images/bears.jpg"

src = imageLoad(src)
cv2.imshow("src", src)
src_not = cv2.bitwise_not(src)
src_not = binaryTo(src_not) # 이진화를 하지않으면 레이블링의 오차범위가 넓어짐

sharp = sharpTo(src)
binary = binaryTo(sharp)
line, fiveline = Findfiveline(binary) # 오선의 좌표값 추출

del_line = delete_line(src, line) # 오선삭제
del_line = binaryTo(del_line) # 이진화작업
del_line = cv2.bitwise_not(del_line)

# 모폴로지 연산을 위한 커널 사각형(2 x 2) 생성, 악보크기에 따라 커널 사각형의 값이 적절해야함
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # 사각형 커널

mophol_img = cv2.morphologyEx(del_line, cv2.MORPH_DILATE, kernal, iterations=1)
# mophol_img = cv2.dilate(del_line, kernal, iterations=1)
mophol_img = cv2.bitwise_not(mophol_img)
cv2.imshow("mo", mophol_img)

first_labeling, label_ary = first_labeling(src_not) # 1차 레이블링 레이블한 범위를 배열에 저장
dst = cv2.bitwise_and(mophol_img, mophol_img, mask = first_labeling) # and연산을 이용해 mophol_img에서 mask부분만 나타냄
first_inv = cv2.bitwise_not(first_labeling) # 배경이미지에 관심영역을 넣기위한 labeling이미지의 inv
add = cv2.add(dst, first_inv) # 배경과 잘라낸 이미지 합성
# cv2.imshow("add", add)

add_inv = cv2.bitwise_not(add)

second_labeling2, label_ary2 = second_labeling(add_inv)
dst2 = cv2.bitwise_and(add, add, mask = second_labeling2)
second_labeling2_inv = cv2.bitwise_not(second_labeling2)
add2 = cv2.add(dst2, second_labeling2_inv)

cv2.imshow("add2", add2)

add2_inv = cv2.bitwise_not(add2)

kernal2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 타원형 커널
kernal3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 사각형 커널

# 열기연산으로 머리와 꼬리를 나눈후 팽창연산으로 음표의 머리를 메꾸고 침식연산으로 머리 외 부분을 완전히 삭제
mophol_img2 = cv2.morphologyEx(add2_inv, cv2.MORPH_OPEN, kernal2, iterations=1)
mophol_img2 = cv2.morphologyEx(mophol_img2, cv2.MORPH_DILATE, kernal3, iterations=1)
mophol_img2 = cv2.morphologyEx(mophol_img2, cv2.MORPH_ERODE, kernal3, iterations=3)

# cv2.imshow("sobel", sobel)
cv2.imshow("mo2", mophol_img2)

third, notes = third_labeling(mophol_img2)
# for i in range(len(notes)):
#     print(notes[i].__dict__)
third.sort()

# 오선의 범위별로 음표의 머리좌표들을 note_heads에 저장
degree = 50
note_heads = []
for i in range(len(fiveline)):
    note_head = []
    if i+1 != len(fiveline):
        degree = int((fiveline[i][4] + fiveline[i+1][0]) / 2)
        degree = degree - fiveline[i][4]
        print(degree)

    for j in range(len(third)):
        if fiveline[i][0] - degree <= third[j][1] <= fiveline[i][4] + degree:
            note_head.append(third[j])
        
    note_heads.append(note_head)


# 좌표들을 x좌표 기준으로 정렬
for i in range(len(note_heads)):
    note_heads[i].sort()
    # print(note_heads[i])

# print(fiveline)
# print(third)
# print(note_heads)

note_names = findnotename(note_heads, fiveline)

# for i in range(len(note_names)):
#     print(note_names[i])

kernel4 = np.array([[0, -1, 0],
                    [0, 2, 0],
                    [0, -1, 0]], dtype = np.uint8)
                    
kernel5 = np.array([[0, 0, -1, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 4, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, -1, 0, 0]], dtype = np.uint8)

# kernel4 = np.dtype(np.unit8)
# kernal4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)) # 사각형 커널
mophol_img3 = cv2.morphologyEx(add2_inv, cv2.MORPH_ERODE, kernel4, iterations=1)
# mophol_img3 = cv2.morphologyEx(mophol_img3, cv2.MORPH_ERODE, kernel5, iterations=1)
cv2.imshow("mophol_img3", mophol_img3)
_, note_beats, notes = print_labeling(mophol_img3, notes)

note_beats.sort()
notes2 = []
for i in range(len(third)):
    note = Note()
    note.set_head(third[i])
    note.set_beat(note_beats[i])
    notes2.append(note)


dist = 50
for i in range(len(fiveline)):
    if i+1 != len(fiveline):
        dist = int((fiveline[i][4] + fiveline[i+1][0]) / 2)
        dist = dist - fiveline[i][4]
        print(dist)

    for j in range(len(notes2)):
        if fiveline[i][0] - degree <= notes2[j].note_head[1] <= fiveline[i][4] + degree:
            notes2[j].set_fline(i+1)

for i in range(len(notes)):
    print(notes2[i].__dict__)


cv2.waitKey(0)
cv2.destroyAllWindows()
