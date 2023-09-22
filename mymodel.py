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
        self.ReLU = nn.LeakyReLU()

    def forward(self, x, M):
        input = x
        feature_map = self.BN1(self.conv1(x))
        x = self.ReLU(feature_map)
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
        
        # cancel the erase model
        # # 生成掩码
        # k = np.random.randint(1, self.MC)
        # ACk = AC[:, k, :, :].detach()
        # # ACk = F.interpolate(ACk.unsqueeze(dim=1), scale_factor=32, mode='bilinear')
        # ACk = self.upsample(ACk.unsqueeze(dim=1))
        # sita = np.random.uniform(0.2, 0.5)
        # zero = torch.zeros_like(ACk)
        # one = torch.ones_like(ACk)
        # mask = torch.where(ACk>sita, zero, one)
        # erase_image = image*mask
        # erase_image = erase_image

        # Ex = self.MiddleLayer(self.LowLayer(erase_image))
        # _, _, Ex = self.lightWeight1(Ex, self.MF)
        # FE, AE, _ = self.lightWeight2(self.HighLayer(Ex), self.MC)

        # return (FF, AF), (FC, AC), (FE, AE)
        return (FF, AF), (FC, AC)

class A2(nn.Module):
    def __init__(self, MF, MC):
        super().__init__()

        self.MC = MC
        self.MF = MF
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()

    def forward(self, AF, AC):
        AC = self.upsample_layer(AC)
        AC = AC.repeat(1, self.MF//self.MC, 1, 1)
        AC = self.sigmoid(AC)
        # print(f"AF'shape{AF.shape}, AC'shape {AC.shape}")
        return AF*AC
    
class LSTM(nn.Module):
    def __init__(self, input_channels, num_hiddens, M):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.M = M
        self.input_channels = input_channels
        # 这里batch_size只能设置4
        self.W_xi = nn.Linear(input_channels, num_hiddens)
        self.W_hi = nn.Linear(num_hiddens, num_hiddens)
        self.Sig_i = nn.Sigmoid()
        self.W_xf = nn.Linear(input_channels, num_hiddens)
        self.W_hf = nn.Linear(num_hiddens, num_hiddens)
        self.Sig_f = nn.Sigmoid()
        self.W_xo = nn.Linear(input_channels, num_hiddens)
        self.W_ho = nn.Linear(num_hiddens, num_hiddens)
        self.Sig_o = nn.Sigmoid()
        self.W_xc = nn.Linear(input_channels, num_hiddens)
        self.W_hc = nn.Linear(num_hiddens, num_hiddens)
        self.tanh = nn.Tanh()

    def forward(self, input, device):
        batch_size = input.shape[0]
        input = input.view(batch_size, self.M, -1) # [batch_size, M, Channels(maybe 512 or 256)]
        input = input.transpose(0, 1)
        (H, C) = (torch.zeros((batch_size, self.num_hiddens), device=device),
            torch.zeros((batch_size, self.num_hiddens), device=device))
        for X in input:
            I = self.Sig_i(self.W_xi(X) + self.W_hi(H))
            F = self.Sig_i(self.W_xf(X) + self.W_hf(H))
            O = self.Sig_i(self.W_xo(X) + self.W_ho(H))
            C_tilda = self.tanh(self.W_xc(X) + self.W_hc(H))
            C = F*C + I*C_tilda
            H = O*self.tanh(C)
        # 返回最后产生的h [batch_size, num_hiddens]

        return H
        


class DFF(nn.Module):
    def __init__(self, num_hiddens, M, beta, feature_size):
        super().__init__()
        self.LSTM = LSTM(feature_size, num_hiddens, M)
        self.beta = beta

    def BAP(self, F, A, device):
        V = torch.zeros((A.shape[0], 1), device=device)
        for i in range(A.shape[1]):
            fk = F*torch.unsqueeze(A[:, i,:,:], dim=1)
            fk = nn.functional.adaptive_avg_pool2d(fk, 1)
            fk = torch.squeeze(fk) # BxC
            # fk = fk.view(1, fk.shape[0], fk.shape[1])
            V = torch.cat((V, fk), dim=1)
        # print(V[:, 1:])
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
        loss = MSE(data.data, delta.detach())
        return loss, C

    def forward(self, F, A, C, device):
        V = self.BAP(F, A, device=device)
        RCloss, C = self.RCLoss(V, C)
        h_M = self.LSTM(V, device)
        return h_M, RCloss, C
        # A = torch.mean(A, dim=1, keepdim=True)
        # feature = F*A
        # feature = nn.functional.adaptive_avg_pool2d(feature, 1)
        # feature = feature.squeeze()
        # # return h_M, C
        # return feature

class CFJLNet(nn.Module):
    def __init__(self, MF, MC, beta, num_hiddens, genderSize):
        super().__init__()
        self.MC = MC
        self.MF = MF
        self.FS_F = 256
        self.FS_C = 512
        self.beta = beta
        self.num_hiddens = num_hiddens
        # self.backbone ,self.out_channels = get_ResNet50()
        self.extrad_feature = FeatureExtract(*get_ResNet50(), self.MF, self.MC)
        # self.featureExtrate = nn.Sequential(*self.backbone[0:8])

        self.A2 = A2(self.MF, self.MC)
        self.Fine_DFF = DFF(num_hiddens=self.num_hiddens, M=self.MF, beta=self.beta, feature_size=self.FS_F)
        self.Coarse_DFF = DFF(num_hiddens=self.num_hiddens, M=self.MC, beta=self.beta, feature_size=self.FS_C)
        self.gender_encoder = nn.Sequential(
            nn.Linear(1, genderSize),
            nn.BatchNorm1d(genderSize),
            nn.ReLU()
        )
        self.Fine_MLP = nn.Sequential(
            nn.Linear(256 + genderSize, 128),
            # nn.Linear(128 + genderSize, 1)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.Coarse_MLP = nn.Sequential(
            nn.Linear(256 + genderSize, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.MLP = nn.Sequential(
        #     nn.Linear(self.out_channels + genderSize, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )

    def forward(self, image, gender, Fine_C, Coarse_C, device):
        # F_data, C_data, E_data = self.extrad_feature(image)
        F_data, C_data = self.extrad_feature(image)
        # gF, _ = F_data
        # gC, _ = C_data
        AF_plus = self.A2(F_data[1], C_data[1])

        gF, Fine_RCloss, Fine_C = self.Fine_DFF(F_data[0], AF_plus, Fine_C, device=device)
        gC, Coarse_RCloss, Coarse_C = self.Coarse_DFF(*C_data, Coarse_C, device=device)
        # gF, Fine_C = self.Fine_DFF(F_data[0], AF_plus, Fine_C, device=device)
        # gC, Coarse_C = self.Coarse_DFF(*C_data, Coarse_C, device=device)
        # gEC, _, _ = self.Coarse_DFF(*E_data, torch.zeros_like(Coarse_C, dtype=torch.float, device=device), device=device)
        # gEC, _ = self.Coarse_DFF(*E_data, torch.zeros_like(Coarse_C, dtype=torch.float, device=device), device=device)
        # gF = self.Fine_DFF(F_data[0], AF_plus, Fine_C, device=device)
        # gC = self.Coarse_DFF(*C_data, Coarse_C, device=device)
        # gEC = self.Coarse_DFF(*E_data, torch.zeros_like(Coarse_C, dtype=torch.float, device=device), device=device)
        gender_feature = self.gender_encoder(gender)

        gF = torch.cat([gF, gender_feature], dim=1)
        gC = torch.cat([gC, gender_feature], dim=1)
        # gEC = torch.cat([gEC, gender_feature], dim=1)
        yF = self.Fine_MLP(gF)
        yC = self.Coarse_MLP(gC)
        # yEC = self.Coarse_MLP(gEC)

        # return yF, yC, yEC, Fine_RCloss, Coarse_RCloss, Fine_C, Coarse_C
        return yF, yC, Fine_RCloss, Coarse_RCloss, Fine_C, Coarse_C
    
    # def forward(self, image, gender):
    #     feature = self.featureExtrate(image)
    #     feature = F.adaptive_avg_pool2d(feature, 1)
    #     feature = torch.squeeze(feature)
    #     gender_feature = self.gender_encoder(gender)
    #     feature = feature.view(-1, self.out_channels)
    #     g = torch.cat([feature, gender_feature], dim=1)
    #     x = self.MLP(g)

    #     return x




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
    # print(net)
    # total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)
    yF, yC, yEC, Fine_RCloss, Coarse_RCloss, Fine_C, Coarse_C = net(image, gender, Fine_C, Coarse_C, torch.device(f'cuda:{0}'))
    # yF, yC, yEC, Fine_C, Coarse_C = net(image, gender, Fine_C, Coarse_C, torch.device(f'cuda:{0}'))
    # print(f"yF.shape:{yF}, \nyC.shape:{yC}, \nyEC.shape:{yEC}\nFine_RCloss = {Fine_RCloss}, Coarse_RCloss = {Coarse_RCloss}\nFine_C.shape:{Fine_C.shape}, Coarse_C.shape:{Coarse_C.shape}")

    # KL_F = F.kl_div(yF.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
    # KL_C = F.kl_div(yC.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
    # KL_EC = F.kl_div(yEC.softmax(dim=-1).log(), label.softmax(dim=-1), reduction='sum')
    # MAE_F = loss_fn(yF, label)
    # MAE_C = loss_fn(yC, label)
    # MAE_EC = loss_fn(yEC, label)
    # loss = ((MAE_F + KL_F) + (MAE_C + KL_C) + (MAE_EC + KL_EC))/3 + Fine_RCloss + Coarse_RCloss
    # loss.backward()
    # print(f"yF'grad is {yF.grad}")
    # print(f"loss is {loss}")

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
