    
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
        return self.name

    def get_beat(self):
        return self.beat
    
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
        raise masktypeWrong # 'alive', 'delete'외의 타입이라면 예외발생

    return src

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

        # 높이가 평균높이 - 10 보다 작다면 온음표
        if roi.shape[0] < avg_height - 10:
            note.set_beat(1)
        else:
            # 폭이 평균보다 크다면 8분음표
            if roi.shape[1] > avg_width + 1:
                note.set_beat(8)
            else:
                center = int(roi.shape[1] / 2)

                for j in range(-1, 1):
                    pre_value = 0 # 이전값
                    change_count = 0 # 변환하는 count
                    sum = 0

                    for i in range(len(roi)):
                        sum += roi[i][center + j]
                        if roi[i][center + j] != pre_value:
                            pre_value = roi[i][center + j]
                            change_count += 1

                    # 변환횟수가 3회 이상이면 2분음표로 추정하고 break
                    # 화소들의 평균이 40미만이면 2분음표 이상이면 8분음표
                    sum = sum / int(len(roi))
                    if change_count >= 3:
                        if sum < 40:
                            note.set_beat(2)
                        else:
                            note.set_beat(8)

                    # 3회 미만이면 4분음표이지만 오차범위를 돌리기 위해 not break
                    else:
                        note.set_beat(4)

# 템플릿 매칭
def templating(src, temp):

    w, h = temp.shape[::-1]
    res = cv2.matchTemplate(src, temp, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9 # 정확도
    loc = np.where(res >= threshold)

    if len(loc[0]) > 0:
        return 1
    else:
        return 0

# 이미지를 불러와 적당한 크기로 resize하기 위한 함수
def resized_img(src):
    '''
    이미지를 리사이즈하는 이유는 이미지가 클수록 오선의 두께가 커지기 때문에 오선을 제거하고
    모폴로지연산중 하나인 팽창연산으로 그 사이를 매꿀때 커널의 크기를 기준하기 어렵다.
    커널의 크기가 너무 크면 2분음표 온음표의 안이 전부 채워지고
    너무 작다면 오선을 제거한 후 공백을 제대로 채워주지 못한다.
    그래서 가장 정확도가 높았던 곰세마리 악보를 기준으로 악보를 리사이즈한다.
    리사이즈할 때의 보간법으로는 쌍 선형 보간법으로 진행했다.(원본을 가장 잘 유지시키기 때문)
    '''
    base_size = 380000 # 38만인 이유는 곰세마리 악보가 383412인데 이 사이즈가 가장 잘 됬으므로 기준으로 잡음
    base_range = int(base_size / 10) # 10%를 오차범위

    while(True):
        if base_size - base_range <= src.size <= base_size + base_range: # 이미지의 사이즈를 오차범위로 비교해서 범위안이라면 리사이즈된 이미지 반환
            return src
        elif src.size > base_size + base_range:
            src = cv2.resize(src, dsize=(0, 0), fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR) # 이미지가 크다면 가로, 세로 0.9배로 리사이즈
        elif src.size < base_size - base_range:
            src = cv2.resize(src, dsize=(0, 0), fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR) # 이미지가 작다면 가로, 세로 1.1배로 리사이즈
    
def find_gill(gill_name):
    if gill_name == '1':
        return imageLoad("./braille_image/1gill.png")
    elif gill_name == '2':
        return imageLoad("./braille_image/2gill.png")
    elif gill_name == '3':
        return imageLoad("./braille_image/3gill.png")
    elif gill_name == '4':
        return imageLoad("./braille_image/4gill.png")
    elif gill_name == '5':
        return imageLoad("./braille_image/5gill.png")
    elif gill_name == '6':
        return imageLoad("./braille_image/6gill.png")
    elif gill_name == '7':
        return imageLoad("./braille_image/7gill.png")
    else:
        print("gill is range out")

def draw_img(x, y, img, output):
    height, width = img.shape
    output[y:y+height, x:x+width] = img

    return height, width

def find_braille_img(note, temp_name):
    if note.get_beat() == 2:
        if temp_name == 'C':
            return imageLoad("./braille_image/C_2.png")
        elif temp_name == 'D':
            return imageLoad("./braille_image/D_2.png")
        elif temp_name == 'E':
            return imageLoad("./braille_image/E_2.png")
        elif temp_name == 'F':
            return imageLoad("./braille_image/F_2.png")
        elif temp_name == 'G':
            return imageLoad("./braille_image/G_2.png")
        elif temp_name == 'A':
            return imageLoad("./braille_image/A_2.png")
        elif temp_name == 'B':
            return imageLoad("./braille_image/B_2.png")

    elif note.get_beat() == 4:
        if temp_name == 'C':
            return imageLoad("./braille_image/C_4.png")
        elif temp_name == 'D':
            return imageLoad("./braille_image/D_4.png")
        elif temp_name == 'E':
            return imageLoad("./braille_image/E_4.png")
        elif temp_name == 'F':
            return imageLoad("./braille_image/F_4.png")
        elif temp_name == 'G':
            return imageLoad("./braille_image/G_4.png")
        elif temp_name == 'A':
            return imageLoad("./braille_image/A_4.png")
        elif temp_name == 'B':
            return imageLoad("./braille_image/B_4.png")

    elif note.get_beat() == 8:
        if temp_name == 'C':
            return imageLoad("./braille_image/C_8.png")
        elif temp_name == 'D':
            return imageLoad("./braille_image/D_8.png")
        elif temp_name == 'E':
            return imageLoad("./braille_image/E_8.png")
        elif temp_name == 'F':
            return imageLoad("./braille_image/F_8.png")
        elif temp_name == 'G':
            return imageLoad("./braille_image/G_8.png")
        elif temp_name == 'A':
            return imageLoad("./braille_image/A_8.png")
        elif temp_name == 'B':
            return imageLoad("./braille_image/B_8.png")

    else:
        raise beatERROR
    
src = "./images/naviya.png"
# src = "./images/bears.jpg"

src = imageLoad(src)
src = resized_img(src)
cv2.imshow("src", src)
src_not = cv2.bitwise_not(src)
src_not = binaryTo(src_not) # 이진화를 하지않으면 레이블링의 오차범위가 넓어짐

# sharp = sharpTo(src)
binary = binaryTo(src)
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
    if label_ary[i]['area'] > 3500: # 넓이가 3500 이상이면 오선을 포함하는 사각형 / 2000으로 기준잡아도 잘됨
        roi_maker(fline_mask, label_ary[i], 'alive')
    else:
        roi_maker(fline_mask, label_ary[i], 'delete')
        

fline_dst = cv2.bitwise_and(mophol_img, mophol_img, mask = fline_mask) # and연산을 이용해 mophol_img에서 mask부분만 나타냄
fline_mask_inv = cv2.bitwise_not(fline_mask) # 배경이미지에 관심영역을 넣기위한 labeling이미지의 inv
fline_add = cv2.add(fline_dst, fline_mask_inv) # 배경과 잘라낸 이미지 합성
fline_add_inv = cv2.bitwise_not(fline_add)

'''
템플릿 매칭을 이용한 박자 인식
템플릿매칭을 한 반환값이 1이라면 해당박자로 인식
base_beat는 박자표의 밑숫자이고 sum_beat는 박자표의 윗숫자이다
밑숫자가 악보의 기준이 되는 박자이기 때문에 기준값으로 저장하고
sum_beat는 기준이 되는 박자의 개수이기 때문에 박자들의 합이 sum_beat라면 한마디가 된다
'''
template_4and4 = imageLoad("./template/4and4.png")
template_4and2 = imageLoad("./template/4and2.jpg")
temp_result = templating(fline_add, template_4and4)
if temp_result == 1:
    base_beat = 4
    sum_beat = 4
    base_beat_img = imageLoad("./braille_image/4and4.png")
else:
    temp_result = templating(fline_add, template_4and2)
    if temp_result == 1:
        base_beat = 4
        sum_beat = 2
        base_beat_img = imageLoad("./braille_image/4and2.png")

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

note_dst = cv2.bitwise_and(fline_add, fline_add, mask = note_mask)
note_mask_inv = cv2.bitwise_not(note_mask)
note_add = cv2.add(note_dst, note_mask_inv)

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
for i in range(int(label_cnt)):
        heads_ary.append([int((label_ary[i]['x']*2+label_ary[i]['width'])/2), int((label_ary[i]['y']*2+label_ary[i]['height'])/2)])
heads_ary.sort()

# 세로모양의 3x3커널 생성(오선밖 음표의 중심에 선이 있을경우 제거하는데 효과적임)
kernel4 = np.array([[0, -1, 0],
                    [0, 2, 0],
                    [0, -1, 0]], dtype = np.uint8)

mophol_img3 = cv2.morphologyEx(note_add_inv, cv2.MORPH_ERODE, kernel4, iterations=1)
cv2.imshow("mophol_img3", mophol_img3)

label_cnt, label_ary = labeling(mophol_img3)
note_rect_ary = []

for i in range(int(label_cnt)):
        note_rect = []
        note_rect.append(label_ary[i]['x'])
        note_rect.append(label_ary[i]['y'])
        note_rect.append(label_ary[i]['width'])
        note_rect.append(label_ary[i]['height'])
        note_rect_ary.append(note_rect)
note_rect_ary.sort()

'''
정렬된 머리좌표들과 음표의 사각형 값들을 넣어줌
사각형의 x좌표값이 겹치면 순서가 엉켜서 버그발생
사각형의 x, y값과 머리좌표의 x, y값의 오차범위를 이용해 해결
'''
notes = []
for i in range(len(note_rect_ary)):
    note = Note()
    for j in range(len(heads_ary)):
        if note_rect_ary[i][0] < heads_ary[j][0] < note_rect_ary[i][0] + 10:
            if note_rect_ary[i][1] < heads_ary[j][1] < note_rect_ary[i][1] + 100:
                note.set_note_head(heads_ary[j])
                note.set_note_rect(note_rect_ary[i])
                notes.append(note)
                break


# 오선의 한 뭉치일때를 위한 초기값
fline_range = 50
for i in range(len(fiveline)):
    if i+1 != len(fiveline): # 마지막 오선이 아니라면 오선의 범위값 조정
        fline_range = int((fiveline[i][4] + fiveline[i+1][0]) / 2)
        fline_range = fline_range - fiveline[i][4]

    for j in range(len(notes)):
        if fiveline[i][0] - fline_range <= notes[j].note_head[1] <= fiveline[i][4] + fline_range:
            notes[j].set_fline(i)

findnotename(fiveline, notes)
findnotebeat(notes, mophol_img3)

notes.sort(key = lambda object:object.fline_area)

for i in range(len(notes)):
    print(notes[i].__dict__)

# 출력을 위한 이미지 생성
output = np.zeros(src.shape, dtype = src.dtype)
output[:] = 255

# 박자표에 대한 점자 이미지에 넣어줌
add_height = 10 # 점자와 점자 사이의 높이거리를 10만큼 떨어트림
x = 0
y = 20
height, width = draw_img(x, y, base_beat_img, output)
y = y + height + add_height # 박자표를 넣고 x=0으로 해야하기 때문에 값 변경하지 않음, y는 이미지의 높이 + add_height

# 길표 삽입
first_gill = notes[0].get_name()
gill_img = find_gill(first_gill[:1])
height, width = draw_img(x, y, gill_img, output)
x = x + width

#도화지 하나 만들고 x, y 기준 잡고 fline 같으면 y = 30, x += 30 if x >= image.cols y+= 50
pre_fline = 0 # 이전의 오선의 영역을 알기위한 변수
width = 0
height = 0
temp_beat = 0
#길, 마디, 마침표
for note in notes:
    temp = None
    temp_name = note.get_name()
    gill = temp_name[:1]
    temp_name = temp_name[1:2]
   
    if note.get_fline() != pre_fline:
        pre_fline += 1
        y += height + add_height
        x = 0
        gill_img = find_gill(gill)
        height, width = draw_img(x, y, gill_img, output)
        x += width

    temp = find_braille_img(note, temp_name) # temp에 음표 이미지 저장

    temp_beat += base_beat / note.get_beat() # 기준이 되는 박자가 분자가 되고 각 음표에 할당된 박자들은 분모가 된다
    height, width = draw_img(x, y, temp, output)

    if sum_beat == int(temp_beat):
        x += width
        temp_beat = 0
        
    x += width

cv2.imshow("output", output)

cv2.waitKey(0)
cv2.destroyAllWindows()