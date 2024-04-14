import torch


def ckpt_convert(input_ckpt: str, output_ckpt: str):
    ckpt_dict = torch.load(input_ckpt, map_location='cpu')
    pass


if __name__ == "__main__":
    ckpt1 = "/mnt/code/ckpts/vit_base_e49_l0.0276.ckpt"
    ckpt2 = "/mnt/code/ckpts/group_e49_l0.0646.ckpt"
    ckpt3 = "/mnt/code/ckpts/meta_e49_l0.0218.ckpt"

    out_ckpt1 = "/mnt/code/ckpts/vit_base_e49_l0.0276_v2.ckpt"
    out_ckpt2 = "/mnt/code/ckpts/group_e49_l0.0646_v2.ckpt"
    out_ckpt3 = "/mnt/code/ckpts/meta_e49_l0.0218_v2.ckpt"

    ckpt_convert(ckpt1, out_ckpt1)