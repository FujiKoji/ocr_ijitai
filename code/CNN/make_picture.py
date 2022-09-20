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

    
    from sklearn.datasets import fetch_openml
import numpy as np
import pandas
import matplotlib.pyplot as plt

#MNISTデータの呼び出し
mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

#RGB値を正規化する
x_all = mnist_X.astype(np.float32).to_numpy() / 255
y_all = mnist_y.astype(np.int32).to_numpy()

#学習データの中身を確認
plt.imshow(x_all[0].reshape(28,28),cmap='gray')

#データローダーの作成
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

#データを訓練とテストに分割
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1/6, random_state=0)

#データをPyTorchのTensorに変換
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#データとラベルをセットにしたDatasetを作成
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)
 
#データセットのミニバッチサイズを指定した、Dataloaderを作成
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=18000, shuffle=False)

#メモリが大きいため必要ない変数を削除
del(x_all)
del(y_all)
del(X_train)
del(X_test)
del(y_train)
del(y_test)
del(ds_test)
del(ds_train)
del(mnist_X)
del(mnist_y)

# 3. ネットワークの構築 
from torch import nn
 
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))
 

# 4. 誤差関数と最適化手法の設定
from torch import optim
# 誤差関数の設定
loss_fn = nn.CrossEntropyLoss()
 
# 重みを学習する際の最適化手法の選択
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 5. 学習と推論の設定
# 5-1. 学習1回でやることを定義
from torch.autograd import Variable
 
def train(epoch):
    model.train()  # ネットワークを学習モードに切り替える
 
    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in loader_train:
        data, target = Variable(data), Variable(target)  # 微分可能に変換
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット
 
        output = model(data)  # 入力dataをinputし、出力を求める
        loss = loss_fn(output, target)  # 出力と訓練データの正解との誤差を求める
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する

    print("epoch{}：終了\n".format(epoch))


# 5-2. 推論1回でやることを定義
def test():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in loader_test:
        data, target = Variable(data), Variable(target)  # 微分可能に変換
        output = model(data)  # 入力dataをinputし、出力を求める
        # 推論する
        pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
        correct += pred.eq(target.data.view_as(pred)).sum()  # 正解と一緒だったらカウントアップ

    # 正解率を出力
    data_num = len(loader_test.dataset)  # データの総数
    print('\nテストデータの正解率: {}/{} ({:.0f}%)\n'.format(correct,data_num, 100. * correct / data_num))


# 6. 学習と推論の実行
for epoch in range(3):
    train(epoch)

if __name__ == '__main__':
    for epock in range(3):
        train(epoch)
    test()
