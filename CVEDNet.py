

import torch
import torch.nn as nn
from models.util.complex_util import ComplexConv1D,ComplexRelu,ComplexBatchNorm1D,ComplexSigmoid,ComplexMaxPool1D
from ptflops import get_model_complexity_info
import pandas as pd
import time
import re

class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim, 1))
        self.bias = nn.Parameter(torch.zeros(self.dim, 1))


    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim):
        super(CumulativeLayerNorm, self).__init__(dim)

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.cat((x.real, x.imag), dim=1)
        x = torch.transpose(x, 1, 2) # 交换第2维和第3维的位置
        x = super().forward(x) # 对通道层进行归一化
        x = torch.transpose(x, 1, 2) # 交换第1维和第2维的位置
        return x




class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(ComplexConv1d, self).__init__()
        self.re_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.im_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if torch.is_complex(x):
                x = x.real , x.imag
            if x.dtype in [torch.float32, torch.float64]:
                x = torch.split(x,x.size(1) // 2, dim=1)
        real_output = self.re_conv(x[0].clone())-self.im_conv(x[1].clone()) # [0]为real，[1]为imag
        imag_output = self.im_conv(x[0].clone())+self.re_conv(x[1].clone())
        return real_output, imag_output

class ComplexTransConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(ComplexTransConv1d, self).__init__()
        self.re_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        self.im_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
    def forward(self, x, **kwargs):
        real_output = self.re_conv(x[0].clone(), **kwargs)-self.im_conv(x[1].clone(), **kwargs) # [0]为real，[1]为imag
        imag_output = self.im_conv(x[0].clone(), **kwargs)+self.re_conv(x[1].clone(), **kwargs)
        return real_output, imag_output


class DilationConv_Block(nn.Module):
    def __init__(self, in_channels=256, out_channels=512, kernel_size=3, dilation=1):
        super(DilationConv_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = ComplexConv1d(in_channels, out_channels, 1) # 1*1卷积调整通道
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = GlobalLayerNorm(out_channels*2)

        self.dropout1 = nn.Dropout(0.2)
        self.pad = (dilation * (kernel_size - 1)) // 2
        self.dwconv = ComplexConv1d(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = GlobalLayerNorm(out_channels*2)
        self.dropout2 = nn.Dropout(0.3)
        self.Sc_conv_1 = ComplexConv1d(out_channels, in_channels, 1, bias=True)
        self.Sc_conv_2 = ComplexConv1d(out_channels, in_channels, 1, bias=True)

    def forward(self, x):
        c0 = self.conv1x1(x)
        c1 = torch.cat([c0[0], c0[1]], dim=1)
        c1 = self.PReLU_1(c1)
        c1 = self.norm_1(c1)
        c1 = self.dropout1(c1)
        c2 = torch.split(c1, c1.size(1) // 2, dim=1)
        c3 = self.dwconv(c2)
        c4 = torch.cat([c3[0], c3[1]], dim=1)
        c4 = self.PReLU_1(c4)
        c4 = self.norm_2(c4)
        c4 = self.dropout2(c4)
        c5 = torch.split(c4, c4.size(1) // 2, dim=1)
        skip = self.Sc_conv_1(c5)
        output = self.Sc_conv_2(c5)
        return output,skip

class Encoder_Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(Encoder_Block, self).__init__()
        # conv 1 x 1
        self.conv1= ComplexConv1D(in_channels, out_channels, 16, stride=16 // 2, padding=0)
        self.conv2 = ComplexConv1D(in_channels, out_channels, 4, stride=4 // 2, padding=0)
        self.pool2 = ComplexMaxPool1D(kernel_size=4, stride=4, padding=0)
        self.conv1x1 = ComplexConv1D(out_channels*2, out_channels,kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        if isinstance(x, tuple) and len(x) == 2:
            # 如果是元组且包含两个元素，拆分并组合成复数
            x = torch.complex(x[0], x[1])
        if not torch.is_complex(x):
            x = torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)
        x1 = self.conv1(x)
        x2_1 = self.conv2(x)
        x2_2 = self.pool2(x2_1)
        x3 = torch.cat((x1 ,x2_2), dim=1, out=None)
        output = self.conv1x1(x3)
        return output
class StackedDBCBs(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan=None,
        n_blocks=8,
        n_repeats=1,
    ):
        super(StackedDBCBs, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.DBCBlock = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                self.DBCBlock.append(
                    DilationConv_Block(in_channels=in_chan, out_channels=out_chan,kernel_size=3,dilation=2**x)
                    )

    def forward(self, output):
        skip_connection = torch.zeros(1, device=output[0].device) , torch.zeros(1, device=output[0].device)
        for layer in self.DBCBlock:
            tcn_out = layer(output)
            residual, skip = tcn_out
            skip_connection_clone = skip_connection[0].clone() ,skip_connection[1].clone()
            skip_connection = skip_connection_clone[0] + skip[0] , skip_connection_clone[1] + skip[1]
            output_clone = output[0].clone() ,output[1].clone()
            output = output_clone[0] + residual[0] , output_clone[1] + residual[1]
        return skip_connection


class CVEDNet(nn.Module):
    def __init__(self,
                 A=512, # encoder输出的通道
                 B=16, # encoder和decoder的卷积核大小
                 C=128, # in_channels StackedDBCB的输入输出通道
                 D=512, # out_channels StackedDBCB的中间通道
                 E=8, # block_num 堆叠的模块
                 F=3, # repeat 重复的组数
                  ):
        super(CVEDNet, self).__init__()
        self.encoder = Encoder_Block()
        self.LayerN=  CumulativeLayerNorm(A*2)
        self.BottleNeck = ComplexConv1d(A, C, 1)
        self.StackedDBCB = StackedDBCBs(in_chan= C ,out_chan=D,n_blocks=E,n_repeats=F)
        self.Expansion = ComplexConv1d(C, A, 1)
        self.decoder = ComplexTransConv1d(A, 1, B, stride=B//2)

    def forward(self, x):
        w0 = self.encoder(x)
        w2 = self.LayerN(w0)
        e = self.BottleNeck(w2)
        e1 = self.StackedDBCB(e)
        m1 = self.Expansion(e1)
        s = self.decoder(m1)
        return s


def calculate_model_params(model: nn.Module):
    total_params = 0
    layer_params = {}

    # 遍历模型的每一层
    for name, param in model.named_parameters():
        # 获取参数的总数
        param_count = param.numel()
        total_params += param_count
        layer_params[name] = param_count

    # 打印每一层的参数数量
    for layer_name, num_params in layer_params.items():
        print(f"Layer {layer_name}: {num_params} parameters")

    # 打印总参数数量
    print(f"Total parameters: {total_params}")
    return total_params


def calculate_model_params_and_flops0(model: nn.Module, input_res=(3, 224, 224)):
    total_params = 0
    layer_params = {}

    # 参数量统计
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        layer_params[name] = param_count

    print("\n----- Parameter Count -----")
    for layer_name, num_params in layer_params.items():
        print(f"Layer {layer_name}: {num_params} parameters")
    print(f"Total parameters: {total_params}\n")

    # FLOPs 计算
    print("----- FLOPs Estimation -----")
    with torch.cuda.device(0):  # 若无GPU可改为 `with torch.device('cpu')`
        macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                                 print_per_layer_stat=True, verbose=False)
        print(f"FLOPs (MACs x2): {macs}")
        print(f"Total Parameters (from ptflops): {params}")

    return total_params, macs

def calculate_model_params_and_flops(model: nn.Module, input_res=(3, 224, 224), n_runs=100):
    total_params = 0
    layer_params = {}

    # 参数量统计
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        layer_params[name] = param_count

    print("\n----- Parameter Count -----")
    for layer_name, num_params in layer_params.items():
        print(f"Layer {layer_name}: {num_params} parameters")
    print(f"Total parameters: {total_params}\n")

    # FLOPs 计算
    print("----- FLOPs Estimation -----")
    with torch.cuda.device(0):  # 若无GPU可改为 `with torch.device('cpu')`
        macs, params = get_model_complexity_info(
            model, input_res, as_strings=True, print_per_layer_stat=True, verbose=False
        )
        print(f"FLOPs (MACs x2): {macs}")
        print(f"Total Parameters (from ptflops): {params}")

    # -------------------------------
    # ✅ 推理时间 & FPS & TFLOPs 计算
    # -------------------------------
    print("\n----- Inference Speed Test (FPS & TFLOPs) -----")

    model.eval()
    dummy_input = torch.randn(1, *input_res).to('cuda')  # 单个样本
    # 预热 CUDA
    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize()  # 确保 CUDA 操作完成
    start = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()

    avg_infer_time = (end - start) / n_runs  # 平均推理时间（秒）
    fps = 1.0 / avg_infer_time  # FPS（每秒推理的帧数）

    # 使用正则表达式提取数字部分
    try:
        match = re.match(r"(\d+(\.\d+)?)\s*(G|M|K)?Mac", macs.strip())  # 匹配 MACs 数值
        if match:
            mac_value = float(match.group(1))  # 获取数字部分
            unit = match.group(3)  # 获取单位部分

            if unit == 'G':
                gmacs = mac_value  # GMacs
            elif unit == 'M':
                gmacs = mac_value / 1000  # 转换为 GMac
            elif unit == 'K':
                gmacs = mac_value / 1e6  # 转换为 GMac
            else:
                gmacs = mac_value / 1e9  # 默认处理为 GMacs
        else:
            raise ValueError(f"Cannot parse MACs: {macs}")
    except ValueError as e:
        print(e)
        return

    total_flops = gmacs * 2  # FLOPs = MACs × 2（乘法 + 加法）
    tflops = (fps * total_flops) / 1000  # 每秒算力需求（TFLOPs）

    print(f"Avg inference time: {avg_infer_time * 1000:.3f} ms")
    print(f"FPS: {fps:.2f} frames per second")
    print(f"Theoretical compute: {tflops:.3f} TFLOPs/s")

    return total_params, macs, fps, tflops


if __name__ == "__main__":
    import netron
    from tensorboardX import SummaryWriter

    real_part = torch.randn(3, 1, 768).to("cuda")
    imag_part = torch.randn(3, 1, 768).to("cuda")
    # 将实部和虚部组合成复数张量
    # input = torch.complex(real_part, imag_part)
    input = torch.cat([real_part, imag_part], dim=1)
    model = CVEDNet().to("cuda")
    # s = model(input)

    # with SummaryWriter(logdir='log2') as w:
    #     w.add_graph(model, input)
    #
    # torch.onnx.export(model, input, f='AlexNet.onnx')  # 导出 .onnx 文件
    # netron.start('AlexNet.onnx')

    # print(model.weight)
    # calculate_model_params(model)
    calculate_model_params_and_flops(model,input_res = (2, 768))
    # calculate_model_params_and_flops()

# 计算卷积参数
    # 4. 实例化模型
    # model1 = Encoder_Block()

    # 5. 统计可训练参数量
    # total_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    # print(f"可训练参数总数: {total_params}")
