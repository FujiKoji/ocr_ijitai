import cnn_method
from cnn_method import Cnn, Execution
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# グラフのスタイルを指定
plt.style.use('seaborn-darkgrid')

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
f.close()

#labelデータの読み込み
f = open(file_path_y,"r",encoding='UTF-8')
data_y = f.readlines() #１行ずつ読み込み
label_list = []
for label in data_y: #「高」:0,「髙」:1 に対応
    if label[0] == "高":
        label_list.append(0)
    elif label[0] == "髙":
        label_list.append(1)
f.close()

loader_train, loader_test, ds_train, ds_test = cnn_method.organaize(img_list_int,label_list)

#インスタンス化
model = Cnn()

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 最適化手法を設定
optimizer = optim.Adam(model.parameters())

execution = Execution(model, optimizer, criterion, loader_train, loader_test)
train_loss_list, test_loss_list = execution.run(model ,30, optimizer, criterion)

num_epochs=30
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(range(num_epochs), train_loss_list, c='b', label='train loss')
ax.plot(range(num_epochs), test_loss_list, c='r', label='test loss')
ax.set_xlabel('epoch', fontsize='20')
ax.set_ylabel('loss', fontsize='20')
ax.set_title('training and test loss', fontsize='20')
ax.grid()
ax.legend(fontsize='25')
plt.show()

#学習したモデルに画像を入力し、可視化
check = DataLoader(ds_test, batch_size=1, shuffle=False)
dataiter = iter(check)
for ii in range(10):
    datas, targets = dataiter.next()
    model.eval()
    prediction = model(datas)
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(np.transpose(datas[0], (1, 2, 0)))  # 各軸の順番を変更
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  
    ax.set_title(f"True: {targets[0].item()}, Prediction: {prediction.argmax().item()}", fontsize=20)
    plt.show()