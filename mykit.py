import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet50

def get_ResNet50():
    # model = resnet50(pretrained = True)
    model = resnet50(pretrained=True)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels

class lightWeight(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.conv1 = backbone[0]
        self.BN1 = backbone[1]
        self.conv2 = backbone[2]
        self.BN2 = backbone[3]
        self.conv3 = backbone[4]
        self.BN3 = backbone[5]
        self.ReLU = backbone[6]

    def forward(self, x, M):
        input = x
        feature_map = self.BN1(self.conv1(x))
        x = feature_map
        x = self.BN2(self.conv2(x))
        attention_map = self.ReLU(x)
        attention_map = attention_map[:, :M, :, :]
        x = self.BN3(self.conv3(x))
        x = self.ReLU(input + x)
        return feature_map, attention_map, x



class FeatureExtract(nn.Module):

    def __init__(self, backbone, output_channles, MF, MC):

        super().__init__()
        self.outChannels = output_channles
        self.MC = MC
        self.MF = MF
        self.LowLayer = nn.Sequential(*backbone[:6]) # (3, H, W) -> (64, H/4, W/4) -> (256, H/4, W/4) ->(512, H/8, W/8)
        self.MiddleLayer = backbone[6][0] # (1024, H/16, W/16) 去除原BTNK3中的后四层
        self.lightWeight1 = lightWeight(list(backbone[6][1].children()))
        self.HighLayer = backbone[7][0] # (2048, H/32, W/32) 去除原BTNK4中的最后一层
        self.lightWeight2 = lightWeight(list(backbone[7][1].children()))

        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, image):
        x = self.MiddleLayer(self.LowLayer(image))
        FF, AF, x = self.lightWeight1(x, self.MF)
        FC, AC, x = self.lightWeight2(self.HighLayer(x), self.MC)

        # 生成掩码
        k = np.random.randint(1, self.MC)
        ACk = AC[:, k, :, :].detach()
        # ACk = F.interpolate(ACk.unsqueeze(dim=1), scale_factor=32, mode='bilinear')
        ACk = self.upsample(ACk.unsqueeze(dim=1))
        sita = np.random.uniform(0.2, 0.5)
        zero = torch.zeros_like(ACk)
        one = torch.ones_like(ACk)
        mask = torch.where(ACk>sita, zero, one)
        erase_image = image*mask
        erase_image = erase_image

        Ex = self.MiddleLayer(self.LowLayer(erase_image))
        _, _, Ex = self.lightWeight1(Ex, self.MF)
        FE, AE, _ = self.lightWeight2(self.HighLayer(Ex), self.MC)

        return (FF, AF), (FC, AC), (FE, AE)

class A2(nn.Module):
    def __init__(self, MF, MC):
        super().__init__()

        self.MC = MC
        self.MF = MF
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, AF, AC):
        AC = self.upsample_layer(AC)
        AC = AC.repeat(1, self.MF//self.MC, 1, 1)
        AC = self.sigmoid(AC)
        # print(f"AF'shape{AF.shape}, AC'shape {AC.shape}")
        return AF*AC
    
class LSTM(nn.Module):
    def __init__(self, num_hiddens, M):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.M = M

    def get_lstm_params(self, feature_length, num_hiddens, device):
        num_inputs = num_outputs = feature_length

        def normal(shape):
            return (torch.randn(size=shape, device=device)*0.01)

        def three():
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xi, W_hi, b_i = three()  # 输入门参数
        W_xf, W_hf, b_f = three()  # 遗忘门参数
        W_xo, W_ho, b_o = three()  # 输出门参数
        W_xc, W_hc, b_c = three()  # 候选记忆元参数
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                b_c, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_lstm_state(self, batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),
                torch.zeros((batch_size, num_hiddens), device=device))
    
    def lstm(self, inputs, state, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
        W_hq, b_q] = params
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H, C)

    def forward(self, input, device):
        batch_size = input.shape[0]
        input = input.view(batch_size, self.M, -1)
        feature_length = input.shape[2]
        input = input.transpose(0, 1)
        state = self.init_lstm_state(batch_size, self.num_hiddens, device=device)
        params = self.get_lstm_params(feature_length, self.num_hiddens, device=device)
        H, (h, c)=self.lstm(input, state, params)
        # 返回最后产生的h [batch_size, num_hiddens]
        return h
        


class DFF(nn.Module):
    def __init__(self, num_hiddens, M, beta):
        super().__init__()
        self.LSTM = LSTM(num_hiddens, M)
        self.beta = beta

    def BAP(self, F, A, device):
        V = torch.zeros((A.shape[0], 1), device=device)
        for i in range(A.shape[1]):
            fk = F*torch.unsqueeze(A[:,i,:,:], dim=1)
            fk = nn.functional.adaptive_avg_pool2d(fk, 1)
            fk = torch.squeeze(fk) # BxC
            V = torch.cat((V, fk), dim=1)
        return V[:, 1:] # BxMC

    def RCLoss(self, data, C):
        # M = 32 或 64，C = 256 或 512
        # data.shape=BxMC, C.shape=1xMC
        # C 的形状应该和V一致 V.shape = [AC, C]
        B = data.shape[0]
        delta = C.repeat(B, 1)
        MSE = nn.MSELoss()
        tmp = torch.sum(data-delta, dim=0)
        C = C + torch.unsqueeze((self.beta*tmp/B), dim=0)
        delta = C.repeat(B, 1)
        loss = MSE(data, delta.detach())
        return loss, C

    def forward(self, F, A, C, device):
        V = self.BAP(F, A, device=device)
        RCloss, C = self.RCLoss(V, C)
        h_M = self.LSTM(V, device=device)
        return h_M, RCloss, C

class CFJLNet(nn.Module):
    def __init__(self, MF, MC, beta, num_hiddens, genderSize):
        super().__init__()
        self.MC = MC
        self.MF = MF
        self.beta = beta
        self.num_hiddens = num_hiddens
        self.extrad_feature = FeatureExtract(*get_ResNet50(), self.MF, self.MC)
        self.A2 = A2(self.MF, self.MC)
        self.Fine_DFF = DFF(num_hiddens=self.num_hiddens, M=self.MF, beta=self.beta)
        self.Coarse_DFF = DFF(num_hiddens=self.num_hiddens, M=self.MC, beta=self.beta)
        self.gender_encoder = nn.Sequential(
            nn.Linear(1, genderSize),
            nn.BatchNorm1d(genderSize),
            nn.ReLU()
        )
        self.Fine_MLP = nn.Sequential(
            nn.Linear(self.num_hiddens + genderSize, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.Coarse_MLP = nn.Sequential(
            nn.Linear(self.num_hiddens + genderSize, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, gender, Fine_C, Coarse_C, device):
        F_data, C_data, E_data = self.extrad_feature(image)
        AF_plus = self.A2(F_data[1], C_data[1])
        gF, Fine_RCloss, Fine_C = self.Fine_DFF(F_data[0], AF_plus, Fine_C, device=device)
        gC, Coarse_RCloss, Coarse_C = self.Coarse_DFF(*C_data, Coarse_C, device=device)
        gEC, _, _ = self.Coarse_DFF(*E_data, torch.zeros_like(Coarse_C, dtype=torch.float, device=device), device=device)
        gender_feature = self.gender_encoder(gender)
        gF = torch.cat([gF, gender_feature], dim=1)
        gC = torch.cat([gC, gender_feature], dim=1)
        gEC = torch.cat([gEC, gender_feature], dim=1)
        yF = self.Fine_MLP(gF)
        yC = self.Coarse_MLP(gC)
        yEC = self.Coarse_MLP(gEC)

        return yF, yC, yEC, Fine_RCloss, Coarse_RCloss, Fine_C, Coarse_C





import matplotlib.pyplot as plt
from matplotlib import cm
if __name__ == '__main__':
    loss_fn = nn.L1Loss(reduction='sum')
    image = torch.randint(0, 10, (10, 3 ,448, 448), dtype=torch.float).cuda()
    label = torch.randint(0, 10, (10, 1), dtype=torch.float).cuda()
    # print(type(image))
    MC = 32
    MF = 64
    beta = 0.95
    Fine_C = torch.zeros((1, MF*256)).cuda()
    Coarse_C = torch.zeros((1, MC*512)).cuda()
    num_hiddens = 128
    genderSize = 32
    gender = torch.randint(0, 2, (10, 1), dtype=torch.float).cuda()
    net = CFJLNet(MF, MC, beta, num_hiddens, genderSize).cuda()
    total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)
    yF, yC, yEC, Fine_RCloss, Coarse_RCloss, Fine_C, Coarse_C = net(image, gender, Fine_C, Coarse_C)
    # print(f"yF.shape:{yF.shape}, yC.shape:{yC.shape}, yEC.shape:{yEC.shape}\nFine_RCloss = {Fine_RCloss}, Coarse_RCloss = {Coarse_RCloss}\nFine_C.shape:{Fine_C.shape}, Coarse_C.shape:{Coarse_C.shape}")
    KL_F = F.kl_div(yF.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
    KL_C = F.kl_div(yC.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
    KL_EC = F.kl_div(yEC.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
    MAE_F = loss_fn(yF, label)
    MAE_C = loss_fn(yC, label)
    MAE_EC = loss_fn(yEC, label)
    loss = ((MAE_F + KL_F) + (MAE_C + KL_C) + (MAE_EC + KL_EC))/3 + Fine_RCloss + Coarse_RCloss
    print(f"loss is {loss}")
    # FF, AF, FC, AC, ACk, k = net(image)
    # FF, AF, FC, AC = net(image)
    # print(f"this is FF's shape: {FF.shape}\nthis is AF's shape: {AF.shape}\nthis is FC's shape: {FC.shape}\nthis is AC's shape:{AC.shape}")
    # bilinear_upsample = F.interpolate(ACk.view(10, 1, 14, 14), scale_factor=32, mode='bilinear')
    # bicubic_upsample = F.interpolate(ACk.view(10, 1, 14, 14), scale_factor=32, mode='bicubic')
    # nearest_upsample = F.interpolate(ACk.view(10, 1, 14, 14), scale_factor=32, mode='nearest')
    
    # sita = np.random.uniform(0.2, 0.5)
    # zero = torch.zeros_like(bicubic_upsample)
    # one = torch.ones_like(bicubic_upsample)
    # mask1 = torch.where(bilinear_upsample > sita, zero, one)
    # mask2 = torch.where(bicubic_upsample > sita, zero, one)
    # mask3 = torch.where(nearest_upsample > sita, zero, one)
    # pics_gray = torch.mean(image, dim=1)
    # erase = torch.mean(erase_pics, dim=1)
    # bilinear_mask = pics_gray[1]*mask1
    # bicubic_mask = pics_gray[1]*mask2
    # nearest_mask = pics_gray[1]*mask3
    # plt.figure()
    # plt.subplot(2,4,1)
    # plt.imshow(pics_gray[1].squeeze().detach().numpy())
    # plt.subplot(2,4,2)
    # plt.imshow(bicubic_upsample[1].squeeze().detach().numpy())
    # plt.subplot(2,4,3)
    # plt.imshow(bilinear_upsample[1].squeeze().detach().numpy())
    # plt.subplot(2,4,4)
    # plt.imshow(nearest_upsample[1].squeeze().detach().numpy())
    # plt.subplot(2,4,5)
    # plt.imshow(pics_gray[1].squeeze().detach().numpy())
    # plt.subplot(2,4,6)
    # plt.imshow(bicubic_mask[1].squeeze().detach().numpy())
    # plt.subplot(2,4,7)
    # plt.imshow(bilinear_mask[1].squeeze().detach().numpy())
    # plt.subplot(2,4,8)
    # plt.imshow(nearest_mask[1].squeeze().detach().numpy())
    # plt.colorbar(cax=None, ax=None, shrink=1)
    # plt.show()

    # pli_pic = []
    # for i in range(pics_gray.shape[0]):
    #     pli_pic.append(pics_gray[i].squeeze())
    # for i in range(mask.shape[0]):
    #     pli_pic.append(mask[i].squeeze())
    # for i in range(erase.shape[0]):
    #     pli_pic.append(erase[i].squeeze())
    # fig, axes = plt.subplots(3, image.shape[0], figsize=(image.shape[0] * 2, 3 * 2))
    # axes = axes.flatten()
    # for i, (ax, img) in enumerate(zip(axes, pli_pic)):
    #     if torch.is_tensor(img):
    #         im = ax.imshow(img.numpy())
    #     else:
    #         # PIL Image
    #         im = ax.imshow(img, cmap='gray')
    # plt.show()
