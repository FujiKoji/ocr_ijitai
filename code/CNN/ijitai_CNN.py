import cnn_method
from cnn_method import input_image
from cnn_method import Cnn, Execution
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# グラフのスタイルを指定
plt.style.use('seaborn-darkgrid')

#作成したファイルの呼び出し
file_path_x = "picture_data/datax_2.dat"
file_path_y = "picture_data/datay_2.dat"
file_path_info = "picture_data/info_2.dat"

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

#文字情報取得
f = open(file_path_info,"r",encoding='UTF-8')
texts = f.read().split(",")

#labelデータの読み込み
f = open(file_path_y,"r",encoding='UTF-8')
data_y = f.readlines() #１行ずつ読み込み
label_list = []
for label in data_y: #「高」:0,「髙」:1 に対応
    for ii in range(len(texts)-1):
        if label[0] == texts[ii]:
            label_list.append(ii)
f.close()

batch_size = 32
test_size = 1/7
loader_train, loader_test, ds_train, ds_test = cnn_method.organaize(img_list_int, label_list, batch_size, test_size)

#インスタンス化
model = Cnn(3)

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 最適化手法を設定
optimizer = optim.Adam(model.parameters())

num_epochs=50
execution = Execution(model, optimizer, criterion, loader_train, loader_test)
train_loss_list, test_loss_list, accuracy_rate_list = execution.run(num_epochs)

#損失推移の可視化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(range(num_epochs), train_loss_list, c='b', label='train loss')
ax.plot(range(num_epochs), test_loss_list, c='r', label='test loss')
ax.set_xlabel('epoch', fontsize='20')
ax.set_ylabel('loss', fontsize='20')
ax.set_title('training and test loss', fontsize='20')
ax.grid()
ax.legend(fontsize='25')
plt.show()

#正解率の可視化
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.plot(range(num_epochs), accuracy_rate_list, c='b')
ax.set_xlabel('epoch', fontsize='20')
ax.set_ylabel('accuracy_rate', fontsize='20')
ax.set_title('accuracy_rate', fontsize='20')
ax.grid()
ax.legend(fontsize='25')
plt.show()

#学習したモデルに画像を入力し、可視化
check = DataLoader(ds_test, batch_size=1, shuffle=False)
dataiter = iter(check)
for ii in range(20):
    datas, targets = dataiter.next()
    model.eval()
    prediction = model(datas)
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(np.transpose(datas[0], (1, 2, 0)))  # 各軸の順番を変更
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  
    ax.set_title(f"True: {targets[0].item()}, Prediction: {prediction.argmax().item()}", fontsize=20)
    plt.show()

#新しい画像の読み取り
# img_file = "write_1.png"
# new_img = input_image(img_file)
# model.eval()
# prediction = model(new_img)
# if prediction.argmax().item() == 0:
#     fig, ax = plt.subplots(dpi=100)
#     ax.imshow(np.array(new_img)[0][0])  # 各軸の順番を変更
#     ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  
#     ax.set_title(f"Prediction: 0", fontsize=20)
#     plt.show()
# elif prediction.argmax().item() == 1:
#     fig, ax = plt.subplots(dpi=100)
#     ax.imshow(np.array(new_img)[0][0])  # 各軸の順番を変更
#     ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  
#     ax.set_title(f"Prediction: 1", fontsize=20)
#     plt.show()