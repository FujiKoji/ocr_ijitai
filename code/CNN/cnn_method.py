#エラーが発生したため追加
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


#画像のリサイズ
def input_image(img_file):
    img = PIL.Image.open(img_file)
    (width, height) = (56,56)
    img_resized = img.resize((width, height))
    gray_img = img_resized.convert('L')
    input_img = np.array([[np.array(gray_img).astype(np.float32) / 255]])
    new_img = torch.Tensor(input_img)

    return new_img

    

# 文字画像データ生成
class MakePicture:
    def __init__(self,font_file, font_size,text):
        self.font_file = font_file
        self.font_size = font_size
        self.text = text
    
    #グレースケールで文字画像を生成
    def gray(self,img_size,width_move,height_move,angle):
        # 画像サイズ，背景色，フォントの色を設定
        canvasSize = (img_size, img_size)
        backgroundL = 256
        textL       = 0

        # 文字を描く画像の作成
        img  = PIL.Image.new('L', canvasSize, backgroundL)
        draw = PIL.ImageDraw.Draw(img)

        # 用意した画像に文字列を描く
        font = PIL.ImageFont.truetype(self.font_file, self.font_size)
        textWidth, textHeight = draw.textsize(self.text,font=font)
        textTopLeft = (canvasSize[0]//2-textWidth//2+width_move, canvasSize[1]//2-textHeight//2+height_move)
        draw.text(textTopLeft, self.text, fill=textL, font=font)
        
        #画像を1次元に変換
        dimention_to_one = np.array(img.rotate(angle,fillcolor=(256))).ravel()
        return dimention_to_one
    
    def binary(self,img_size):
        # 画像サイズ，背景色，フォントの色を設定
        canvasSize = (img_size, img_size)
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
        dimention_to_one_list = []
        for ii in range(len(dimention_to_one)):
            if dimention_to_one[ii] <= 127:
                dimention_to_one_list.append(1)
            else:
                dimention_to_one_list.append(0)
        dimention_to_one_int = np.array(dimention_to_one_list)
        return dimention_to_one_int
    
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

#CNNの実装
def organaize(img_list_int,label_list, batch_size, test_size):
    x = np.array(img_list_int).astype(np.float32) / 255
    y_all = np.array(label_list).astype(np.float32)

    #56*56に変更
    x_all = []
    for ii in range(len(x)):
        x_all.append([x[ii].reshape(56,56)])
    x_all = np.array(x_all)

    #データを訓練とテストに分割
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size, random_state=10)

    #データをPyTorchのTensorに変換
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    #データとラベルをセットにしたDatasetを作成
    ds_train = TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test, y_test)

    #データセットのミニバッチサイズを指定した、Dataloaderを作成
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return loader_train, loader_test, ds_train, ds_test

class Cnn(nn.Module):
    def __init__(self, text_num):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)  # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
        #出力画像サイズ52
        self.pool = nn.MaxPool2d((2, 2))  # プーリング層:（領域のサイズ, ストライド）
        #出力画像サイズ26
        self.conv2 = nn.Conv2d(8, 16, 5)
        #出力画像サイズ11
        self.fc1 = nn.Linear(16*11*11, 128)  # 全結合層
        self.dropout = nn.Dropout(p=0.5)  # ドロップアウト:(p=ドロップアウト率)
        self.fc2 = nn.Linear(128, text_num)
    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = x2.view(-1, 16*11*11)
        x4= F.relu(self.fc1(x3))
        x5 = self.dropout(x4)
        x6 = self.fc2(x5)
        return x6

class Execution:
    def __init__(self, model, optimizer, criterion, loader_train, loader_test):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loader_train = loader_train
        self.loader_test = loader_test

    def train(self):
        train_loss = 0
        self.model.train()
        for data, target in self.loader_train:
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(self.loader_train.dataset)
        return train_loss

    def test(self):
        self.model.eval()
        test_loss=0
        correct = 0
        with torch.no_grad():
            for data, target in self.loader_test:
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss = self.criterion(output, target)
                test_loss += loss.item()
            test_loss = test_loss / len(self.loader_test.dataset)
        data_num = len(self.loader_test.dataset)  # データの総数
        print('\nテストデータの正解率: {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))
        accuracy_rate = 100. * correct / data_num
        return test_loss, accuracy_rate

    def run(self, num_epochs):
        train_loss_list = []
        test_loss_list = []
        accuracy_rate_list = []
        for epoch in range(num_epochs):
            train_loss = self.train()
            test_loss, accuracy_rate = self.test()
            print(f'Epoch [{epoch+1}], train_Loss : {train_loss:.4f}, test_Loss : {test_loss:.4f}')
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            accuracy_rate_list.append(accuracy_rate)
        return train_loss_list, test_loss_list, accuracy_rate_list