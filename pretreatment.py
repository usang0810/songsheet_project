import cv2
from PIL import Image, ImageFilter

sheetname = "saveme.png"
src = "./images/"

original = Image.open(src + sheetname)
# print(original.size)
resize_image = original.resize((512, 512))
# print(resize_image.size)
sheetname2 = sheetname.split(".")
resizedname = sheetname2[0] + '_resized.' + sheetname2[1]
resize_image.save(src + resizedname)

original_sheet = cv2.imread(src + resizedname, cv2.IMREAD_COLOR)

gray_sheet = cv2.cvtColor(original_sheet, cv2.COLOR_RGB2GRAY)
ret, dst = cv2.threshold(gray_sheet, 225, 255, cv2.THRESH_BINARY)

cv2.imshow("original", original_sheet)
cv2.imshow("gray_sheet", dst)
cv2.waitKey(0)

# import cv2

# sheetname = "saveme.png"
# src = "./images/"
# size = (512, 512)

# original_sheet = cv2.imread(src + sheetname, cv2.IMREAD_COLOR)

# gray_sheet = cv2.cvtColor(original_sheet, cv2.COLOR_RGB2GRAY)

# resized_sheet = cv2.resize(gray_sheet, size, interpolation = cv2.INTER_CUBIC)


# cv2.imshow("resize", resized_sheet)
# cv2.waitKey(0)
# cv2.destroyAllWindows()