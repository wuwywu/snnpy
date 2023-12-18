import torch
# 一些全局的设定
snnpy = r"E:\snnpy\snnpy"               # snnpy的路径
# 选择运行设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 设定全局数据精度（用64位）
torch.set_default_dtype(torch.float64)  
# torch.set_default_dtype(torch.float32)  
# 全局禁用梯度计算
torch.set_grad_enabled(False)

dt = 0.01            # Integration time step
