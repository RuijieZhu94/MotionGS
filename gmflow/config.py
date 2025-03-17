from yacs.config import CfgNode as CN
_CN = CN()


_CN.feature_channels = 128
_CN.num_scales = 1
_CN.upsample_factor = 8
_CN.num_head = 1
_CN.attention_type = 'swin'
_CN.ffn_dim_expansion = 4
_CN.num_transformer_layers = 6
_CN.model = './gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth'

def get_cfg():
    return _CN.clone()