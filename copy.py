import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

class Note:
    def __init__(self):
        self.note_head = None
        self.note_rect = None
        self.fline_area = 0
        self.name = None
        self.beat = None
        
    def set_note_head(self, note_head):
        self.note_head = note_head

    def set_note_rect(self, note_rect):
        self.note_rect = note_rect

    def set_fline(self, fline_area):
        self.fline_area = fline_area

    def set_name(self, name):
        self.name = name

    def set_beat(self, beat):
        self.beat = beat

    def get_note_head(self):
        return self.note_head

    def get_note_rect(self):
        return self.note_rect

    def get_note_rect_element(self):
        return self.note_rect[0], self.note_rect[1], self.note_rect[2], self.note_rect[3]

    def get_fline(self):
        return self.fline_area
        
    def get_name(self):
        return self.fline_name

    def get_beat(self):
        return self.fline_beat
    
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

# labeling을 할때는 흑백전환 후 해야함
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
        note.set_note_head([int((x*2+width)/2), int((y*2+height)/2)])

        notes.append(note)
        # print("label_array: ", label_ary[i-1])
        
    return label_ary, notes

# 이미지의 각 요소들을 라벨링
def labeling(src):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
    stats_ary = [] # 딕셔너리 정보들을 담을 배열

    for i in range(1, int(cnt)): # cnt : 라벨링 갯수, 라벨링 첫번째 영역은 이미지 전체이므로 제외
        x, y, width, height, area = stats[i]
        dic = {'x' : x, 'y' : y, 'width' : width, 'height' : height, 'area' : area}
        stats_ary.append(dic)

    return cnt-1, stats_ary # cnt - 1 하는 이유는 반복문의 시작을 1부터 했기 때문

# 흑색 이미지에 roi영역 흰색 사각형 만들기 타입이 반대라면 삭제
def roi_maker(src, label, mask_type = 'alive'):
    if mask_type == 'alive':
        cv2.rectangle(src, (label['x'], label['y']), (label['x'] + label['width'], label['y'] + label['height']), 255, -1)
    elif mask_type == 'delete':
        cv2.rectangle(src, (label['x'], label['y']), (label['x'] + label['width'], label['y'] + label['height']), 0, -1)
    else:
        raise mask_typeWrong # 'alive', 'delete'외의 타입이라면 예외발생

    return src

# # 템플릿 매칭
# def templating(src, temp):

#     w, h = temp.shape[::-1]
#     res = cv2.matchTemplate(src, temp, cv2.TM_CCOEFF_NORMED)
#     # print(res.shape)

#     # print("def templating: ", res)

#     threshold = 0.90 # 정확도
#     loc = np.where(res >= threshold)
#     # print("def loc templating: ", loc)
#     ary = [] # 좌표값들
#     for pt in zip(*loc[::-1]):
#         ary2 = [pt]
#         bottom_right = pt[0] + w, pt[1] + h
#         ary2.append(bottom_right)
#         # ary.append([(pt[0]*2+w)/2, (pt[1]*2+h)/2])
#         ary.append(ary2)
        
#         cv2.rectangle(src, pt, bottom_right, 0, 1)
#         # print("top_left", pt)
#         # print("bottom_right", bottom_right)
#         # cv2.circle(src, (int((pt[0]*2+w)/2), int((pt[1]*2+h)/2)), 5, 0, 2)

#     return src, ary

# 템플릿 디렉토리에서 파일 가져오기
def search(dirname):
    filenames = os.listdir(dirname)
    ary = []

    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ary.append(full_filename)
        # print (full_filename)

    return ary

# 계이름 검출
def findnotename(fiveline, notes):
    '''
    오선의 가장 밑에 걸쳐있는 계이름은 4옥타브 '미'이다
    오선과 오선을 뺀 값에서 나누기 2를 하면 각 계이름이 위치할 수 있는 값이 나오는데
    이 값을 이용하여 계이름을 추출한다
    추출하기 위해 계이름을 전부 담은 배열을 선언하고 4옥타브 '도'부터 시작하여 찾는다
    '''
    note_name_def = ['4C', '4D', '4E', '4F', '4G', '4A', '4B', '5C', '5D', '5E', '5F', '5E', '5F', '5G', '5A', '5B',
 '6C',' 6D', '6E', '6F', '6E', '6G']

    for note in notes:
        line_area = note.get_fline()
        dist = int((fiveline[line_area][1] - fiveline[line_area][0]) / 2)

        line_value = fiveline[line_area][4] + (dist*2)

        for i in range(len(note_name_def)):
            if line_value - 1 <= note.note_head[1] <= line_value +1:
                note.set_name(note_name_def[i])
                break
            else:
                line_value -= dist

# 박자 검출
def findnotebeat(notes, image):
    '''
    음표의 영역의 평균값을 이용하여 1, 2, 4, 8분 음표 구분
    가로가 평균보다 길다면 8분음표, 세로가 평균보다 작다면 온음표
    8, 16, 32.. 이후의 음표들은 먼저 8분음표로 저장 후 Note의 beat가 8인 경우에만
    반복문을 이용해 8, 16, 32분 음표를 세분화 하는 작업이 필요
    그 외의 온음표, 2, 4분음표는 바로 구분
    2분음표와 4분음표 열의 중심값(오차범위 : -1~+1)을 이용하여
    이전값과 다른값이 들어왔을 때 count를 증가시켜 count가 3이상이라면
    2분음표로 구분하고 아니라면 4분음표로 구분한다
    '''
    avg_width = int()
    avg_height = int()
    for note in notes:
        x, y, width, height = note.get_note_rect_element()
        roi = image[y:y+height, x:x+width]
        avg_width += roi.shape[1]
        avg_height += roi.shape[0]

    avg_width = int(avg_width / len(notes))
    avg_height = int(avg_height / len(notes))

    # print(avg_width, avg_height)

    for note in notes:
        x, y, width, height = note.get_note_rect_element()
        roi = image[y:y+height, x:x+width]

        # 높이가 평균높이보다 작다면 온음표
        if roi.shape[0] < avg_height:
            note.set_beat(1)
        else:
            # 폭이 평균보다 크다면 8분음표
            if roi.shape[1] > avg_width:
                note.set_beat(8)
            else:
                center = int(roi.shape[1] / 2)
                for j in range(-1, 1):
                    pre_value = 0 # 이전값
                    change_count = 0 # 변환하는 count
                    for i in range(len(roi)):
                        if roi[i][center + j] != pre_value:
                            pre_value = roi[i][center + j]
                            change_count += 1

                    # 변환횟수가 3회 이상이면 2분음표로 추정하고 break
                    if change_count >= 3:
                        note.set_beat(2)
                        break
                    # 3회 미만이면 4분음표이지만 오차범위를 돌리기 위해 not break
                    else:
                        note.set_beat(4)



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
mophol_img = cv2.bitwise_not(mophol_img)
# cv2.imshow("mo", mophol_img)

# 오선의 영역들을 mask처리
fline_mask = np.zeros(src.shape, dtype = src.dtype) # 영역에 흰색 사각형을 그리기 위한 검은 배경
label_cnt, label_ary = labeling(src_not)
for i in range(int(label_cnt)):
    if label_ary[i]['area'] > 3500: # 넓이가 3500 이상이면 오선을 포함하는 사각형
        roi_maker(fline_mask, label_ary[i], 'alive')
    else:
        roi_maker(fline_mask, label_ary[i], 'delete')
        

fline_dst = cv2.bitwise_and(mophol_img, mophol_img, mask = fline_mask) # and연산을 이용해 mophol_img에서 mask부분만 나타냄
fline_mask_inv = cv2.bitwise_not(fline_mask) # 배경이미지에 관심영역을 넣기위한 labeling이미지의 inv
fline_add = cv2.add(fline_dst, fline_mask_inv) # 배경과 잘라낸 이미지 합성

fline_add_inv = cv2.bitwise_not(fline_add)

# 음표 부분들만 mask처리
note_mask = np.zeros(src.shape, dtype = src.dtype)
label_cnt, label_ary = labeling(fline_add_inv)
for i in range(int(label_cnt)):
    if 100 <= label_ary[i]['area'] <= 200:
        # mean()함수를 이용해서 영역의 평균값 저장
        label_avg = cv2.mean(fline_add_inv[label_ary[i]['y']:label_ary[i]['y'] + label_ary[i]['height'], label_ary[i]['x']:label_ary[i]['x'] + label_ary[i]['width']])
        # print(label_avg)
        if label_avg[0] < 150: # 이미지의 평균값이 150이하이면 흰색 사각형 그림, 0번째 인덱스에 값이 있음, 150보다 크다면 어떻게 할꺼?????
            roi_maker(note_mask, label_ary[i])
    else:
        roi_maker(note_mask, label_ary[i], mask_type='delete')

# cv2.imshow("note_mask", note_mask)

note_dst = cv2.bitwise_and(fline_add, fline_add, mask = note_mask)
note_mask_inv = cv2.bitwise_not(note_mask)
note_add = cv2.add(note_dst, note_mask_inv)

# cv2.imshow("note_add", note_add)

note_add_inv = cv2.bitwise_not(note_add)

kernal2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 타원형 커널
kernal3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 사각형 커널

# 열기연산으로 머리와 꼬리를 나눈후 팽창연산으로 음표의 머리를 메꾸고 침식연산으로 머리 외 부분을 완전히 삭제
# 모폴로지 연산은 원래 악보에서 흑백 반전 후 연산 진행
mophol_img2 = cv2.morphologyEx(note_add_inv, cv2.MORPH_OPEN, kernal2, iterations=1)
mophol_img2 = cv2.morphologyEx(mophol_img2, cv2.MORPH_DILATE, kernal3, iterations=1)
mophol_img2 = cv2.morphologyEx(mophol_img2, cv2.MORPH_ERODE, kernal3, iterations=3)

cv2.imshow("mo2", mophol_img2)


label_cnt, label_ary = labeling(mophol_img2)
heads_ary = []
notes_ary = []
for i in range(int(label_cnt)):
        heads_ary.append([int((label_ary[i]['x']*2+label_ary[i]['width'])/2), int((label_ary[i]['y']*2+label_ary[i]['height'])/2)])
        note = Note()
        note.set_note_head([int((label_ary[i]['x']*2+label_ary[i]['width'])/2), int((label_ary[i]['y']*2+label_ary[i]['height'])/2)])
        notes_ary.append(note)

heads_ary.sort()

# 오선의 범위별로 음표의 머리좌표들을 fline_note_heads에 저장
fline_range = 50 # 오선의 한 뭉치일때를 위한 초기값
fline_note_heads = []
for i in range(len(fiveline)):
    fline_note_head = []
    if i+1 != len(fiveline): # 마지막 오선이 아니라면 오선의 범위값 조정
        fline_range = int((fiveline[i][4] + fiveline[i+1][0]) / 2)
        fline_range = fline_range - fiveline[i][4]

    for j in range(len(heads_ary)):
        if fiveline[i][0] - fline_range <= heads_ary[j][1] <= fiveline[i][4] + fline_range:
            fline_note_head.append(heads_ary[j])
        
    fline_note_heads.append(fline_note_head)

# print(fline_note_heads)
# 좌표들을 x좌표 기준으로 정렬
for i in range(len(fline_note_heads)):
    fline_note_heads[i].sort()


third, notes = third_labeling(mophol_img2)
third.sort()

# 오선의 범위별로 음표의 머리좌표들을 note_heads에 저장
degree = 50
note_heads = []

for i in range(len(fiveline)):
    note_head = []
    if i+1 != len(fiveline):
        degree = int((fiveline[i][4] + fiveline[i+1][0]) / 2)
        degree = degree - fiveline[i][4]
        # print(degree)

    for j in range(len(third)):
        if fiveline[i][0] - degree <= third[j][1] <= fiveline[i][4] + degree:
            note_head.append(third[j])
        
    note_heads.append(note_head)


# 좌표들을 x좌표 기준으로 정렬
for i in range(len(note_heads)):
    note_heads[i].sort()


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
mophol_img3 = cv2.morphologyEx(note_add_inv, cv2.MORPH_ERODE, kernel4, iterations=1)
# mophol_img3 = cv2.morphologyEx(mophol_img3, cv2.MORPH_ERODE, kernel5, iterations=1)
cv2.imshow("mophol_img3", mophol_img3)




_, note_beats, notes = print_labeling(mophol_img3, notes)

note_beats.sort()
notes2 = []
for i in range(len(third)):
    note = Note()
    note.set_note_head(third[i])
    note.set_note_rect(note_beats[i])
    notes2.append(note)


dist = 50
for i in range(len(fiveline)):
    if i+1 != len(fiveline):
        dist = int((fiveline[i][4] + fiveline[i+1][0]) / 2)
        dist = dist - fiveline[i][4]
        # print(dist)

    for j in range(len(notes2)):
        if fiveline[i][0] - dist <= notes2[j].note_head[1] <= fiveline[i][4] + dist:
            notes2[j].set_fline(i)

findnotename(fiveline, notes2)
findnotebeat(notes2, mophol_img3)

# for i in range(len(notes2)):
#     print(notes2[i].__dict__)

cv2.waitKey(0)
cv2.destroyAllWindows()
