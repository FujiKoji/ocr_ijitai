import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import matplotlib.pyplot as plt

# 使うフォント，サイズ，描くテキストの設定
class MakePicture:
    def __init__(self,font_file, font_size,text):
        self.font_file = font_file
        self.font_size = font_size
        self.text = text
    
    #グレースケールで文字画像を生成
    def gray(self):
        # 画像サイズ，背景色，フォントの色を設定
        canvasSize = (28, 28)
        backgroundL = 256
        textL       = 0

        # 文字を描く画像の作成
        img  = PIL.Image.new('L', canvasSize, backgroundL)
        draw = PIL.ImageDraw.Draw(img)

        # 用意した画像に文字列を描く
        font = PIL.ImageFont.truetype(self.font_file, self.font_size)
        textWidth, textHeight = draw.textsize(self.text,font=font)
        textTopLeft = (canvasSize[0]//2-textWidth//2, canvasSize[1]//2-textHeight//2)
        draw.text(textTopLeft, self.text, fill=textL, font=font)
        
        #画像を1次元に変換
        dimention_to_one = np.array(img).ravel()

        return dimention_to_one

    #datファイル作成
    def write_to_dat(self,file_path_x,file_path_y, dimention_to_one):
        #画像データをdatファイルに書き込み
        f=open(file_path_x,"a")
        for data in dimention_to_one:
            f.write(str(data))
            f.write(" ")
        f.write("\n")
        f.close()

        #正解データをdatファイルに書き込み
        f=open(file_path_y,"a")
        f.write(str(self.text))
        f.write("\n")
        f.close()