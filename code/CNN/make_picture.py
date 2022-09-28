from cnn_method import MakePicture
import numpy as np
import matplotlib.pyplot as plt
import glob

#windowsの場合
font_files = glob.glob("C:/Users/job/Documents/python_code/ocr/Fonts/*")
#macの場合
# font_files = glob.glob("/Users/fujitakouji/Documents/Fonts/*")

# font_size_list = [20,21,22,23,24,25,26,27]
font_size_list = [27]
text_list = ["高","髙"]
file_path_x = "picture_data/datax_2.dat"
file_path_y = "picture_data/datay_2.dat"
file_path_info = "picture_data/info_2.dat"

f=open(file_path_info,"a")
for text in text_list:
    f.write(text+",")
f.close()

for font_file in font_files:
    for text in text_list:
        for font_size in font_size_list:
            make_train = MakePicture(font_file,font_size,text)
            dimention_to_one = make_train.gray()
            make_train.write_to_dat(file_path_x,file_path_y, dimention_to_one)

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

#読み込んだ画像を可視化
fig, axes = plt.subplots(3, 15)
for ii in range(16):
    axes[ii,0].imshow(img_list_int[ii*3].reshape(28,28),cmap='gray')
    axes[ii,1].imshow(img_list_int[ii*3+1].reshape(28,28),cmap='gray')
    axes[ii,1].imshow(img_list_int[ii*3+2].reshape(28,28),cmap='gray')
    
plt.show()
 