import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

class ComplexConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,dilation=1 ,para=None, **kwargs):
        super(ComplexConv1D, self).__init__()
        self.para = para
        if dilation==1:
            if padding==0: self.padding = (kernel_size-1)//2
            else: self.padding = padding
        else:
            self.padding = "same" #int((dilation * (kernel_size - 1)-math.log2(dilation)) // 2)# (dilation * (kernel_size - 1)-1) // 2 #  (dilation * (kernel_size - 1)) // 2
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs)

        if self.para is not None:
            self.init_weights()


    def forward(self, x):
        real_part = self.real_conv(x.real.detach()) - self.imag_conv(x.imag.detach())
        imag_part = self.real_conv(x.imag.detach()) + self.imag_conv(x.real.detach())
        return real_part+1j*imag_part

    def init_weights(self):
        # 根据init_type选择初始化方法
        if self.para == 'kaiming':
            nn.init.kaiming_normal_(self.real_conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.imag_conv.weight, mode='fan_in', nonlinearity='relu')
        elif self.para == 'xavier':
            nn.init.xavier_normal_(self.real_conv.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.xavier_normal_(self.imag_conv.weight, gain=nn.init.calculate_gain('tanh'))
        elif self.para == 'fourier':
            fourier_value = self.fourier_basis(self.real_conv.kernel_size[0], self.real_conv.out_channels)
            self.real_conv.weight = nn.Parameter(fourier_value.real)
            self.imag_conv.weight = nn.Parameter(fourier_value.imag)
            self.real_conv.weight.requires_grad = False
            self.imag_conv.weight.requires_grad = False
            if self.real_conv.bias is not None:
                self.real_conv.bias.requires_grad = False
                self.imag_conv.bias.requires_grad = False
        else:
            raise ValueError("Unsupported initialization type")
        # if self.real_conv.bias is not None:
        #     nn.init.constant_(self.real_conv.bias, 0)
        # if self.imag_conv.bias is not None:
        #     nn.init.constant_(self.imag_conv.bias, 0)

    def fourier_basis(self, kernel_size,out_channels):
        # 生成傅里叶基矩阵
        # t = np.linspace(0, 2 * np.pi, kernel_size)
        # basis = np.array([np.cos(freq * t) for freq in range(num_basis)] +
        #                  [np.sin(freq * t) for freq in range(num_basis)])
        # basis = torch.tensor(basis, dtype=torch.float32)
        # FFT_weights:(kernel_size, out_channels)
        FFT_weights = torch.zeros((kernel_size, out_channels), dtype = torch.complex64)
        for h in range(0, kernel_size):
            for j in range(0, out_channels):
            # 计算复数指数的幂
                exponent = -1j * 2 * torch.pi * (j * h / out_channels)
                # 将计算转换为Tensor
                exponent_tensor = torch.tensor(exponent, dtype=torch.complex64)
                # 使用torch.exp计算并赋值
                FFT_weights[h][j] = torch.exp(exponent_tensor)
                # FFT_weights[h][j] = torch.exp(-1j * 2 * np.pi *(j*h/out_channels))
        # 转换成 conv_weight 的维度：（out_channels, in_channels, kernel_size）
        conv_weight = FFT_weights.permute(1, 0).unsqueeze(1)
        return conv_weight

class ComplexTransConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(ComplexTransConv1d, self).__init__()
        self.re_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.im_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
    def forward(self, x, **kwargs):
        real_output = self.re_conv(x.real, **kwargs)-self.im_conv(x.imag, **kwargs) # [0]为real，[1]为imag
        imag_output = self.im_conv(x.real, **kwargs)+self.re_conv(x.imag, **kwargs)
        return real_output+1j*imag_output

class ComplexSigmoid(nn.Module):
    def __init__(self):
        super(ComplexSigmoid, self).__init__()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.cat((x.real, x.imag), dim=1)
        x2 = self.Sigmoid(x1)
        x3 = torch.split(x2,x2.shape[1] // 2, dim=1)
        return x3[0]+1j*x3[1]
class ComplexRelu(nn.Module):
    def __init__(self):
        super(ComplexRelu, self).__init__()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x1 = torch.cat((x.real, x.imag), dim=1)
        x2 = self.ReLU(x1)
        x3 = torch.split(x2,x2.shape[1] // 2, dim=1)
        return x3[0]+1j*x3[1]

class ComplexDroupout(nn.Module):
    def __init__(self,p, **kwargs):
        super(ComplexDroupout, self).__init__()
        self.Dropout = nn.Dropout(p,**kwargs)

    def forward(self, x):
        x1 = torch.cat((x.real, x.imag), dim=1)
        x2 = self.Dropout(x1)
        x3 = torch.split(x2,x2.shape[1] // 2, dim=1)
        return x3[0]+1j*x3[1]

class ComplexBatchNorm1D(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm1D, self).__init__()
        self.bn = nn.BatchNorm1d(num_features*2)

    def forward(self, x):
        
        x1 = torch.cat((x.real, x.imag), dim=1)
        x2 = self.bn(x1)
        x3 = torch.split(x2,x2.shape[1] // 2, dim=1)
        return x3[0]+1j*x3[1]


class ComplexMaxPool1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2,padding = None):
        super(ComplexMaxPool1D, self).__init__()
        if padding ==  None :
            self.padding = self.padding = (kernel_size-1)//2
        else:
            self.padding = padding
        self.MaxPool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride,padding=self.padding)

    def forward(self, x):
        x1 = torch.cat((x.real, x.imag), dim=1)
        x2 = self.MaxPool1d(x1)
        x3 = torch.split(x2,x2.shape[1] // 2, dim=1)
        return x3[0]+1j*x3[1]


# 当时为了反卷积时向量长度匹配用的，现在用不上了
def compute_convtranspose1d_params(L_in, L_out, kernel_size, stride):
    """
    计算反卷积所需的填充和输出填充。

    参数：
    L_in (int): 输入长度
    L_out (int): 期望输出长度
    kernel_size (int): 卷积核大小
    stride (int): 步幅

    返回：
    tuple: (padding, output_padding)
    """
    if L_in <= 0 or L_out <= 0 or kernel_size <= 0 or stride <= 0:
        raise ValueError("输入长度、输出长度、卷积核大小和步幅必须为正整数。")

    # 计算需要的padding来满足输出长度
    padding = ((L_in - 1) * stride - L_out + kernel_size) // 2
    output_padding = ((L_in - 1) * stride - L_out + kernel_size) % 2
    
    # 确保padding和output_padding非负
    if padding < 0 or output_padding < 0:
        raise ValueError("计算的padding或output_padding为负，请检查输入参数。")

    return padding, output_padding























