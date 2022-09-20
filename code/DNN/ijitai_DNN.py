from sklearn.datasets import fetch_openml
import numpy as np
import pandas
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.autograd import Variable
 

#作成したファイルの呼び出し
file_path_x = "picture_data/datax_1.dat"
file_path_y = "picture_data/datay_1.dat"

#imgデータ読み込み
f = open(file_path_x,"r")
data_x = f.readlines() #１行ずつ読み込み
img_list_str = []
for img in data_x:
    img_list_str.append(img.split(" ")[0:-1])
img_list_int = []
for img in img_list_str:
    img_list_int.append(np.array(list(map(int, img))))

#labelデータの読み込み
f = open(file_path_y,"r",encoding='UTF-8')
data_y = f.readlines() #１行ずつ読み込み
label_list = []
for label in data_y: #「高」:0,「髙」:1 に対応
    if label[0] == "高":
        label_list.append(0)
    elif label[0] == "髙":
        label_list.append(1)


#L値を正規化する
x_all = np.array(img_list_int).astype(np.float32) / 255
y_all = np.array(label_list).astype(np.float32)

#データを訓練とテストに分割
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1/7, random_state=10)

#データをPyTorchのTensorに変換
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#データとラベルをセットにしたDatasetを作成
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

#データセットのミニバッチサイズを指定した、Dataloaderを作成
loader_train = DataLoader(ds_train, batch_size=32, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

# 3. ネットワークの構築 
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))
 

# 4. 誤差関数と最適化手法の設定
# 誤差関数の設定
loss_fn = nn.CrossEntropyLoss()
 
# 重みを学習する際の最適化手法の選択
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 5. 学習と推論の設定
# 5-1. 学習1回でやることを定義
def train(epoch,loss_list):
    model.train()  # ネットワークを学習モードに切り替える
    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in loader_train:
        data, target = Variable(data), Variable(target)  # 微分可能に変換
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット
 
        output = model(data)  # 入力dataをinputし、出力を求める
        loss = loss_fn(output, target)  # 出力と訓練データの正解との誤差を求める
        loss_list = np.append(loss_list,loss.item())
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する

    print("epoch{}：終了\n".format(epoch))
    return loss_list


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
if __name__ == '__main__':
    loss_list = np.arange(0)
    for epoch in range(50):
        loss_list = train(epoch,loss_list)
    plt.plot(loss_list)
    plt.show()
    test()