import torch
import torch.nn as nn
import myKit
import warnings
warnings.filterwarnings("ignore")

""""具体训练参数设置"""

# loss = nn.CrossEntropyLoss(reduction="none")
loss_fn = nn.L1Loss(reduction='sum')

# criterion = nn.CrossEntropyLoss(reduction='none')
if __name__ == '__main__':

    # print(type(image))
    MC = 32
    MF = 64
    beta = 0.95
    Fine_C = torch.zeros((1, MF*256))
    Coarse_C = torch.zeros((1, MC*512))
    num_hiddens = 256
    cls_weight = 2
    net = myKit.get_net(MF, MC, beta, num_hiddens)
    lr = 5e-4
    batch_size = 32
    num_epochs = 50
    weight_decay = 0.0001
    lr_period = 10
    lr_decay = 0.5
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    train_df, valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, 256)
    train_set, val_set = myKit.create_data_loader(train_df, valid_df)
    torch.set_default_tensor_type('torch.FloatTensor')
    # myKit.map_fn(net=net, train_dataset=train_set, valid_dataset=val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay,loss_fn=loss_fn, batch_size=batch_size, model_path="model.pth", record_path="RECORD.csv")
    myKit.map_fn(net, train_set, val_set, num_epochs, lr, weight_decay, lr_period, lr_decay, loss_fn, cls_weight, Fine_C, Coarse_C, batch_size=batch_size, model_path="./model.pth", record_path="./RECORD.csv")
