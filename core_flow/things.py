from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.8
_CN.max_flow = 400
_CN.batch_size = 1
_CN.sum_freq = 100
_CN.val_freq = 5000000
_CN.summary_freq_loss = 1000
_CN.img_freq = 1000
_CN.image_size = [320, 640]
_CN.add_noise = True
_CN.critical_params = []

_CN.transformer = 'latentcostformer'
_CN.restore_ckpt = 'checkpoints/things_kitti.pth'
_CN.data_root = '/mnt/nas/kaichen/flyingthings'

#######################################
_CN.latentcostformer = CN()
_CN.latentcostformer.pe = 'linear'
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.cost_heads_num = 1
# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.context_concat = False
_CN.latentcostformer.encoder_depth = 2
_CN.latentcostformer.feat_cross_attn = False
_CN.latentcostformer.nat_rep = "abs"
_CN.latentcostformer.patch_size = 8
_CN.latentcostformer.patch_embed = 'single'
_CN.latentcostformer.no_pe = False
_CN.latentcostformer.gma = "GMA"
_CN.latentcostformer.kernel_size = 9
_CN.latentcostformer.rm_res = True
_CN.latentcostformer.vert_c_dim = 64
_CN.latentcostformer.cost_encoder_res = True
_CN.latentcostformer.cnet = 'twins'
_CN.latentcostformer.fnet = 'twins'
_CN.latentcostformer.only_global = False
_CN.latentcostformer.add_flow_token = True
_CN.latentcostformer.use_mlp = False
_CN.latentcostformer.vertical_conv = False

# decoder
_CN.latentcostformer.decoder_depth = 6
_CN.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
#_CN.trainer.num_steps = 120000
_CN.trainer.num_steps = 60000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()

import time
import os
import shutil

def process_transformer_cfg(cfg):
    log_dir = ''
    if 'critical_params' in cfg:
        critical_params = [cfg[key] for key in cfg.critical_params]
        for name, param in zip(cfg["critical_params"], critical_params):
            log_dir += "{:s}[{:s}]".format(name, str(param))

    return log_dir

def process_cfg(cfg):
    log_dir = 'logs/' + cfg.name + '/' + cfg.transformer + '/'
    """
    critical_params = [cfg.trainer[key] for key in cfg.critical_params]
    for name, param in zip(cfg["critical_params"], critical_params):
        log_dir += "{:s}[{:s}]".format(name, str(param))

    log_dir += process_transformer_cfg(cfg[cfg.transformer])
    """
    now = time.localtime()
    now_time = '{:02d}_{:02d}_{:02d}_{:02d}_{:02d}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    #log_dir += cfg.suffix + '(' + now_time + ')'
    log_dir += cfg.suffix + now_time
    cfg.log_dir = log_dir
    print('Were going to save log here:', log_dir)
    os.makedirs(log_dir)

    shutil.copytree('configs', f'{log_dir}/configs')
    shutil.copytree('core/FlowFormer', f'{log_dir}/FlowFormer')
