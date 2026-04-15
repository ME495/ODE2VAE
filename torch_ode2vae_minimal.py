import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils import data
from torch.distributions import MultivariateNormal, Normal, kl_divergence as kl
from torch_bnn import BNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchdiffeq import odeint

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from multiprocessing import Process, freeze_support
torch.multiprocessing.set_start_method('spawn', force=True)


# 准备数据集：这里只是一个最小封装，让 DataLoader 能按索引取出一条序列
class Dataset(data.Dataset):
    def __init__(self, Xtr):
        self.Xtr = Xtr # 形状约为 [N, T, 784]
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        return self.Xtr[idx]

# 读取旋转 MNIST 序列数据，并整理成 [样本数, 时间步, 通道, 高, 宽]
mat = loadmat('rot-mnist-3s.mat')
if 'X' in mat:
    X = np.squeeze(mat['X'])
elif 'Xtr' in mat:
    # Some exported variants store the training set directly as [1, N, T, D].
    X = np.squeeze(mat['Xtr'])
else:
    raise KeyError(
        f"Expected one of ['X', 'Xtr'] in rot-mnist-3s.mat, found "
        f"{sorted(k for k in mat.keys() if not k.startswith('__'))}"
    )

if X.ndim != 3:
    raise ValueError(f"Expected data with shape [N, T, D], got {X.shape}")

N = 500
T = 16
Xtr   = torch.tensor(X[:N],dtype=torch.float32).view([N,T,1,28,28])
Xtest = torch.tensor(X[N:],dtype=torch.float32).view([-1,T,1,28,28])
# DataLoader 配置
params = {'batch_size': 25, 'shuffle': True, 'num_workers': 2}
trainset = Dataset(Xtr)
trainset = data.DataLoader(trainset, **params)
testset  = Dataset(Xtest)
testset  = data.DataLoader(testset, **params)

# 一些简单的形状变换模块，方便塞进 nn.Sequential
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.w = w
    def forward(self, input):
        nc = input[0].numel()//(self.w**2)
        return input.view(input.size(0), nc, self.w, self.w)


# ODE2VAE 主体：
# 1) 用卷积编码器把第 0 帧编码到潜变量初值 z0
# 2) 用神经 ODE 在潜空间中沿时间推进，得到 z(t)
# 3) 用解码器把隐状态还原为图像序列
class ODE2VAE(nn.Module):
    def __init__(self, n_filt=8, q=8):
        super(ODE2VAE, self).__init__()
        h_dim = n_filt*4**3 # 编码器输出为 [4*n_filt, 4, 4]，展平后长度即 h_dim
        # 编码器：输入单帧图像，输出紧凑特征
        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )

        # 两个线性层分别给出潜变量初值分布的均值和 log 方差
        self.fc1 = nn.Linear(h_dim, 2*q)
        self.fc2 = nn.Linear(h_dim, 2*q)
        # 解码前先把低维潜变量投回卷积特征空间
        self.fc3 = nn.Linear(q, h_dim)
        # 微分方程右端项 f(v,s)：
        # 这里使用贝叶斯神经网络，既能建模动力学，也能给出函数不确定性
        # 如果想退化成确定性动力学，可设 bnn=False 且 self.beta=0.0
        self.bnn = BNN(2*q, q, n_hid_layers=2, n_hidden=50, act='celu', layer_norm=True, bnn=True)
        # BNN 权重 KL 项的缩放系数；网络过参数化时适当减小会更稳定
        self.beta = 1.0 # 2*q/self.bnn.kl().numel()
        # 解码器：把潜空间中的“位置”状态还原回图像
        self.decoder = nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )
        self._zero_mean = torch.zeros(2*q).to(device)
        self._eye_covar = torch.eye(2*q).to(device) 
        self.mvn = MultivariateNormal(self._zero_mean, self._eye_covar)

    def ode2vae_rhs(self,t,vs_logp,f):
        # 连同 log-density 一起推进，这是连续归一化流里常见的写法
        vs, logp = vs_logp # N,2q & N
        q = vs.shape[1]//2
        dv = f(vs) # “速度”的导数/加速度项，形状 N,q
        ds = vs[:,:q]  # “位置”的导数就是当前速度，形状 N,q
        dvs = torch.cat([dv,ds],1) # 拼成完整状态的导数，形状 N,2q
        # 计算雅可比迹，用于更新 log-density
        ddvi_dvi = torch.stack(
                    [torch.autograd.grad(dv[:,i],vs,torch.ones_like(dv[:,i]),
                    retain_graph=True,create_graph=True)[0].contiguous()[:,i]
                    for i in range(q)],1) # N,q --> df(x)_i/dx_i, i=1..q
        tr_ddvi_dvi = torch.sum(ddvi_dvi,1) # N
        return (dvs,-tr_ddvi_dvi)

    def elbo(self, qz_m, qz_logv, zode_L, logpL, X, XrecL, Ndata, qz_enc_m=None, qz_enc_logv=None):
        ''' Input:
                qz_m        - latent means [N,2q]
                qz_logv     - latent logvars [N,2q]
                zode_L      - latent trajectory samples [L,N,T,2q]
                logpL       - densities of latent trajectory samples [L,N,T]
                X           - input images [N,T,nc,d,d]
                XrecL       - reconstructions [L,N,T,nc,d,d]
                Ndata       - number of sequences in the dataset (required for elbo
                qz_enc_m    - encoder density means  [N*T,2*q]
                qz_enc_logv - encoder density variances [N*T,2*q]
            Returns:
                likelihood
                prior on ODE trajectories KL[q_ode(z_{0:T})||N(0,I)]
                prior on BNN weights
                instant encoding term KL[q_ode(z_{0:T})||q_enc(z_{0:T}|X_{0:T})] (if required) 
        '''
        [N,T,nc,d,d] = X.shape
        L = zode_L.shape[0]
        q = qz_m.shape[1]//2
        # 轨迹先验：把 ODE 采样出的 z(t) 与标准高斯先验比较
        log_pzt = self.mvn.log_prob(zode_L.contiguous().view([L*N*T,2*q])) # L*N*T
        log_pzt = log_pzt.view([L,N,T]) # L,N,T
        kl_zt   = logpL - log_pzt  # L,N,T
        kl_z    = kl_zt.sum(2).mean(0) # N
        kl_w    = self.bnn.kl().sum()
        # 重建似然：这里按 Bernoulli 图像似然计算
        XL = X.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        lhood_L = torch.log(1e-3+XrecL)*XL + torch.log(1e-3+1-XrecL)*(1-XL) # L,N,T,nc,d,d
        lhood = lhood_L.sum([2,3,4,5]).mean(0) # N
        if qz_enc_m is not None: # 额外使用逐时刻编码器约束 ODE 轨迹
            qz_enc_mL    = qz_enc_m.repeat([L,1])  # L*N*T,2*q
            qz_enc_logvL = qz_enc_logv.repeat([L,1])  # L*N*T,2*q
            mean_ = qz_enc_mL.contiguous().view(-1) # L*N*T*2*q
            std_  = 1e-3+qz_enc_logvL.exp().contiguous().view(-1) # L*N*T*2*q
            qenc_zt_ode = Normal(mean_,std_).log_prob(zode_L.contiguous().view(-1)).view([L,N,T,2*q])
            qenc_zt_ode = qenc_zt_ode.sum([3]) # L,N,T
            inst_enc_KL = logpL - qenc_zt_ode
            inst_enc_KL = inst_enc_KL.sum(2).mean(0) # N
            return Ndata*lhood.mean(), Ndata*kl_z.mean(), kl_w, Ndata*inst_enc_KL.mean()
        else:
            return Ndata*lhood.mean(), Ndata*kl_z.mean(), kl_w

    def forward(self, X, Ndata, L=1, inst_enc=False, method='dopri5', dt=0.1):
        ''' Input
                X          - input images [N,T,nc,d,d]
                Ndata      - number of sequences in the dataset (required for elbo)
                L          - number of Monta Carlo draws (from BNN)
                inst_enc   - whether instant encoding is used or not
                method     - numerical integration method
                dt         - numerical integration step size 
            Returns
                Xrec_mu    - reconstructions from the mean embedding - [N,nc,D,D]
                Xrec_L     - reconstructions from latent samples     - [L,N,nc,D,D]
                qz_m       - mean of the latent embeddings           - [N,q]
                qz_logv    - log variance of the latent embeddings   - [N,q]
                lhood-kl_z - ELBO   
                lhood      - reconstruction likelihood
                kl_z       - KL
        '''
        # 训练时的主前向过程：
        # 1) 用首帧 x0 推断初始潜变量分布 q(z0|x0)
        # 2) 在潜空间中用神经 ODE 推进，得到整条轨迹 z(0:T)
        # 3) 解码为图像序列，计算 ELBO = 重建项 - KL 项

        # 输入序列 X: [N,T,nc,d,d]
        # N: batch 大小，T: 时间步，nc: 通道数，d: 图像边长
        [N,T,nc,d,d] = X.shape

        # 只编码第 0 帧，得到初值分布参数（与 many-to-one 设定一致）
        h = self.encoder(X[:,0])
        qz0_m, qz0_logv = self.fc1(h), self.fc2(h) # N,2q & N,2q
        # 潜状态按 [v,s] 拼接：前 q 维近似“速度”，后 q 维近似“位置/内容”
        q = qz0_m.shape[1]//2

        # 重参数化采样 z0 = mu + eps * sigma
        # 这样既能采样又可对 mu/logv 反向传播（VAE 标准做法）
        eps   = torch.randn_like(qz0_m)  # N,2q
        z0    = qz0_m + eps*torch.exp(qz0_logv) # N,2q
        # 对应采样点在标准高斯下的初始 log-density，用于后续流密度累计
        logp0 = self.mvn.log_prob(eps) # N 

        # 构造时间网格并在潜空间积分，得到整条轨迹
        t  = dt * torch.arange(T,dtype=torch.float).to(z0.device)
        ztL   = []
        logpL = []

        # Monte Carlo 采样 L 条轨迹：
        # 每次从 BNN 抽样一个动力学函数 f，反映参数不确定性
        for l in range(L):
            f       = self.bnn.draw_f() # draw a differential function
            # 右端项会同时推进状态 z 和其 log-density
            oderhs  = lambda t,vs: self.ode2vae_rhs(t,vs,f) # make the ODE forward function
            zt,logp = odeint(oderhs,(z0,logp0),t,method=method) # T,N,2q & T,N
            # 调整维度后收集：zt -> [1,N,T,2q], logp -> [1,N,T]
            ztL.append(zt.permute([1,0,2]).unsqueeze(0)) # 1,N,T,2q
            logpL.append(logp.permute([1,0]).unsqueeze(0)) # 1,N,T
        ztL   = torch.cat(ztL,0) # L,N,T,2q
        logpL = torch.cat(logpL) # L,N,T

        # 解码仅使用后半状态 s(t)（内容子空间），而非完整 [v,s]
        st_muL = ztL[:,:,:,q:] # L,N,T,q
        # 合并 L/N/T 三个维度以批量解码，再 reshape 回序列形状
        s = self.fc3(st_muL.contiguous().view([L*N*T,q]) ) # L*N*T,h_dim
        Xrec = self.decoder(s) # L*N*T,nc,d,d
        Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d

        # 计算 ELBO：
        # elbo = lhood - kl_z - beta * kl_w               (默认)
        # elbo = lhood - kl_z - inst_KL - beta * kl_w     (启用 instant encoding)
        if inst_enc:
            # 对每个时刻 x_t 再编码一次，用于约束 ODE 轨迹贴近逐时刻编码分布
            h = self.encoder(X.contiguous().view([N*T,nc,d,d]))
            qz_enc_m, qz_enc_logv = self.fc1(h), self.fc2(h) # N*T,2q & N*T,2q
            lhood, kl_z, kl_w, inst_KL = \
                self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec, Ndata, qz_enc_m, qz_enc_logv)
            elbo = lhood - kl_z - inst_KL - self.beta*kl_w
        else:
            lhood, kl_z, kl_w = self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec, Ndata)
            elbo = lhood - kl_z - self.beta*kl_w

        # 返回重建、潜变量统计量、潜轨迹以及训练监控指标
        return Xrec, qz0_m, qz0_logv, ztL, elbo, lhood, kl_z, self.beta*kl_w

    def mean_rec(self, X, method='dopri5', dt=0.1):
        [N,T,nc,d,d] = X.shape
        # 该函数用于“确定性重建”：
        # 1) 只取首帧 x0 得到潜变量初值均值 z0_mean
        # 2) 用 BNN 的均值动力学在潜空间做 ODE 积分
        # 3) 将每个时刻的内容状态 s(t) 解码回图像

        # 编码第 0 帧，得到特征 h: [N, h_dim]
        h = self.encoder(X[:,0])
        # 初始潜变量均值 q(z0|x0) 的参数: [N, 2q]
        # 其中前 q 维可看作“速度”分量，后 q 维可看作“位置/内容”分量
        qz0_m = self.fc1(h) # N,2q
        q = qz0_m.shape[1]//2

        # 仅推进均值路径，不再像训练 forward 那样联合推进 log-density
        # 状态写作 vs=[v,s]，其导数为 [dv/dt, ds/dt] = [f(v,s), v]
        def ode2vae_mean_rhs(t,vs,f):
            q = vs.shape[1]//2
            dv = f(vs) # N,q: 由神经网络给出的动力学项
            ds = vs[:,:q]  # N,q: 位置导数等于当前速度
            return torch.cat([dv,ds],1) # N,2q

        # 从 BNN 取“均值函数”而非随机采样函数，保证重建稳定可复现
        f     = self.bnn.draw_f(mean=True)
        odef  = lambda t,vs: ode2vae_mean_rhs(t,vs,f)
        # 构造积分时刻: [0, dt, 2dt, ..., (T-1)dt]
        t     = dt * torch.arange(T,dtype=torch.float).to(qz0_m.device)
        # ODE 输出默认为 [T,N,2q]，转置为 [N,T,2q]
        zt_mu = odeint(odef,qz0_m,t,method=method).permute([1,0,2])

        # 仅取后半部分 s(t) 作为解码输入，逐时刻重建图像
        st_mu = zt_mu[:,:,q:] # N,T,q
        # 先展平时间维，统一走一次全连接 + 解码器，再还原回序列形状
        s = self.fc3(st_mu.contiguous().view([N*T,q])) # N*T,h_dim
        Xrec_mu = self.decoder(s) # N*T,nc,d,d
        Xrec_mu = Xrec_mu.view([N,T,nc,d,d]) # N,T,nc,d,d

        # 用像素均方误差衡量重建质量（主要用于测试日志/可视化监控）
        mse = torch.mean((Xrec_mu-X)**2)
        return Xrec_mu,mse
        
# 可视化原图与重建图，默认每次只画前 10 个序列
def plot_rot_mnist(X, Xrec, show=False, fname='rot_mnist.png'):
    N = min(X.shape[0],10)
    Xnp = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    T = X.shape[1]
    plt.figure(2,(T,3*N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+1)
            plt.imshow(np.reshape(Xnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+T+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.savefig(fname)
    if show is False:
        plt.close()


if __name__ == '__main__':
    freeze_support()
    # 构建模型并开始训练
    ode2vae = ODE2VAE(q=8,n_filt=16).to(device)
    Nepoch = 500
    optimizer = torch.optim.Adam(ode2vae.parameters(),lr=1e-3)
    for ep in range(Nepoch):
        # 训练前半程用较少采样降低开销，后半程增加 Monte Carlo 采样提升估计质量
        L = 1 if ep<Nepoch//2 else 5
        for i,local_batch in enumerate(trainset):
            minibatch = local_batch.to(device)
            elbo, lhood, kl_z, kl_w = ode2vae(minibatch, len(trainset), L=L, inst_enc=True, method='rk4')[4:]
            tr_loss = -elbo
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            print('Iter:{:<2d} lhood:{:8.2f}  kl_z:{:<8.2f}  kl_w:{:8.2f}'.\
                format(i, lhood.item(), kl_z.item(), kl_w.item()))
        with torch.set_grad_enabled(False):
            for test_batch in testset:
                test_batch = test_batch.to(device)
                # 每轮用测试集中的一个 batch 做可视化与误差评估
                Xrec_mu, test_mse = ode2vae.mean_rec(test_batch, method='rk4')
                plot_rot_mnist(test_batch, Xrec_mu, False, fname='rot_mnist.png')
                torch.save(ode2vae.state_dict(), 'ode2vae_mnist.pth')
                break
        print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f}  test_mse:{:5.3f}\n'.format(ep, Nepoch, tr_loss.item(), test_mse.item()))
