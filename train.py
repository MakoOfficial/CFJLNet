import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import d2l.torch as d2l


def map_fn(net, train_dataset, valid_dataset, num_epochs, lr, wd, lr_period, lr_decay, loss_fn, cls_weight, Fine_C, Coarse_C, batch_size=32, model_path="./model.pth", record_path="./RECORD.csv"):
    """将训练函数和验证函数杂糅在一起的垃圾函数"""
    devices = d2l.try_all_gpus()
    # 数据读取
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    # 增加多卡训练
    net = nn.DataParallel(net, device_ids=devices)

    ## Network, optimizer, and loss function creation
    net = net.to(devices[0])

    # loss_fn = nn.MSELoss(reduction = 'sum')
    # loss_fn = nn.L1Loss(reduction='sum')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # 每过10轮，学习率降低一半
    scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=lr_period, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    seed=101
    torch.manual_seed(seed)  

    ## Trains

    for epoch in range(num_epochs):
        net.train()
        for batch_idx, data in enumerate(train_loader):
            # #put data to GPU
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).to(devices[0]), gender.type(torch.FloatTensor).to(devices[0])

            batch_size = len(data[1])
            label = data[1].to(devices[0])

            # zero the parameter gradients
            optimizer.zero_grad()
            yF, yC, yEC, Fine_RCloss, Coarse_RCloss, Fine_C, Coarse_C = net(image, gender, Fine_C, Coarse_C, device=devices[0])
            KL_F = F.kl_div(yF.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
            KL_C = F.kl_div(yC.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
            KL_EC = F.kl_div(yEC.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
            MAE_F = loss_fn(yF, label)
            MAE_C = loss_fn(yC, label)
            MAE_EC = loss_fn(yEC, label)
            loss = ((MAE_F + KL_F) + (MAE_C + KL_C) + (MAE_EC + KL_EC))/cls_weight + Fine_RCloss + Coarse_RCloss
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(net, model_path)