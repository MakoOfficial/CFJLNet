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
    loss_fn = nn.L1Loss(reduction='sum')
    # print(type(image))
    MC = 32
    MF = 64
    beta = 0.95
    Fine_C = torch.zeros((1, MF*256))
    Coarse_C = torch.zeros((1, MC*512))
    num_hiddens = 128
    cls_weight = 3
    genderSize = 32
    gender = torch.randint(0, 2, (10, 1), dtype=torch.float)
    net = myKit.get_net(MF=MF, MC=MC, beta=beta, num_hiddens=num_hiddens, genderSize=genderSize)
    lr = 3e-5
    batch_size = 4
    num_epochs = 40
    weight_decay = 0
    lr_period = 4
    lr_decay = 0.1
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    train_df, valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, 128)
    train_set, val_set = myKit.create_data_loader(train_df, valid_df)
    torch.set_default_tensor_type('torch.FloatTensor')
    # myKit.map_fn(net=net, train_dataset=train_set, valid_dataset=val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay,loss_fn=loss_fn, batch_size=batch_size, model_path="model.pth", record_path="RECORD.csv")
    myKit.map_fn(net, train_set, val_set, num_epochs, lr, weight_decay, lr_period, lr_decay, loss_fn, cls_weight, Fine_C, Coarse_C, batch_size=batch_size, model_path="./model.pth", record_path="./RECORD.csv")

        