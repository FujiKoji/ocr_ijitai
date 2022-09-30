from cnn_method import MakePicture
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

#windowsの場合
# ttfontname = "C:\\Windows\\Fonts\\YUMIN.TTF"
#macの場合
font_files = glob.glob("/Users/fujitakouji/Documents/Fonts_IPamj/*")
font_size_list = []
for ii in range(40,50,2):
    font_size_list.append(ii)
# text_list = ["高","髙"] #data_1
text_list = ["榮", "栄", "荣"] #data_2
file_path_x = "picture_data/datax_2.dat"
file_path_y = "picture_data/datay_2.dat"
file_path_info = "picture_data/info_2.dat"
f = open(file_path_info, "a")
for text in text_list:
    f.write(text+",")
f.close()

img_size = 56
count=0
for font_file in font_files:
    for text in text_list:
        for font_size in font_size_list:
            for width_move in range(-5,6,2):
                for height_move in range(-5,6,2):
                    for angle in range(-5,6,2):
                        count+=1
                        print(count)
                        make_train = MakePicture(font_file,font_size,text)
                        dimention_to_one = make_train.gray(img_size,width_move,height_move,angle)
                        make_train.write_to_dat(file_path_x,file_path_y, dimention_to_one)
                        img = dimention_to_one.reshape(img_size,img_size).astype("uint8")
                        kernel = np.ones((2,2),np.uint8)
                        erosion = cv2.erode(img,kernel,iterations = 1)
                        dimention_to_one_erode = erosion.ravel().astype("int64")
                        make_train.write_to_dat(file_path_x,file_path_y, dimention_to_one_erode)

#作成したファイルの呼び出し
f = open(file_path_x,"r")
data_x = f.readlines() #１行ずつ読み込み
img_list_str = []
for img in data_x:
    img_list_str.append(img.split(" ")[0:-1])
img_list_int = []
for img in img_list_str:
    img_list_int.append(np.array(list(map(int, img))))
f.close()


# #読み込んだ画像を可視化
fig, axes = plt.subplots(11,11)
for ii in range(11):
    for jj in range(11):
        axes[ii,jj].imshow(img_list_int[121+ii*10+jj].reshape(img_size,img_size),cmap='gray')
plt.show()
