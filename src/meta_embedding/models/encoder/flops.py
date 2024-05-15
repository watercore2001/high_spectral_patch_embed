from fvcore.nn import FlopCountAnalysis
import torch


def vit_base(x, file):
    from .vit import ViTBase
    model = ViTBase()

if __name__ == "__main__":
    x = torch.randn(10, 96, 96)
    flops = FlopCountAnalysis(model, x)

    with open("output.txt", "w") as f:
        # 使用print将内容输出到文件中
        print("Hello, World!", file=f)
        print("This is a line written to a file.", file=f)