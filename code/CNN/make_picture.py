from cnn_method import MakePicture
import numpy as np
import matplotlib.pyplot as plt
import glob

#windowsの場合
# ttfontname = "C:\\Windows\\Fonts\\YUMIN.TTF"
#macの場合
font_files = glob.glob("/Users/fujitakouji/Documents/Fonts/*")
font_size_list = [20,21,22,23,24,25,26,27]
text_list = ["高","髙"]
file_path_x = "picture_data/datax_1.dat"
file_path_y = "picture_data/datay_1.dat"
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
fig, axes = plt.subplots(16, 16)
for ii in range(16):
    axes[ii,0].imshow(img_list_int[ii*4].reshape(28,28),cmap='gray')
    axes[ii,1].imshow(img_list_int[ii*4+1].reshape(28,28),cmap='gray')
    axes[ii,2].imshow(img_list_int[ii*4+2].reshape(28,28),cmap='gray')
    axes[ii,3].imshow(img_list_int[ii*4+3].reshape(28,28),cmap='gray')
    axes[ii,4].imshow(img_list_int[ii*4+4].reshape(28,28),cmap='gray')
    axes[ii,5].imshow(img_list_int[ii*4+5].reshape(28,28),cmap='gray')
    axes[ii,6].imshow(img_list_int[ii*4+6].reshape(28,28),cmap='gray')
    axes[ii,7].imshow(img_list_int[ii*4+7].reshape(28,28),cmap='gray')
    axes[ii,8].imshow(img_list_int[ii*4+8].reshape(28,28),cmap='gray')
    axes[ii,9].imshow(img_list_int[ii*4+9].reshape(28,28),cmap='gray')
    axes[ii,10].imshow(img_list_int[ii*4+10].reshape(28,28),cmap='gray')
    axes[ii,11].imshow(img_list_int[ii*4+11].reshape(28,28),cmap='gray')
    axes[ii,12].imshow(img_list_int[ii*4+12].reshape(28,28),cmap='gray')
    axes[ii,13].imshow(img_list_int[ii*4+13].reshape(28,28),cmap='gray')
    axes[ii,14].imshow(img_list_int[ii*4+14].reshape(28,28),cmap='gray')
    axes[ii,15].imshow(img_list_int[ii*4+15].reshape(28,28),cmap='gray')

 