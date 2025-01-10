# Reference: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange
from .vision_lstm import ViLBlock
from mamba_ssm.models.mixer_seq_simple import _init_weights

# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False, 
    ):
    embed_dim = cfg['model_cfg']['embed_dim'] # 16
    expansion_factor = cfg['model_cfg']['expansion_factor'] # 4

    block = ViLBlock(
                    dim=embed_dim, #update all these
                    drop_path=0,
                    expansion=expansion_factor, # 2,3 or 4
                    conv_kind="causal1d",
                    conv_kernel_size=4,
                    proj_bias=True,
                    norm_bias=True,
                    seqlens=None,
                    num_blocks=None,
                    init_weights="original",
                )

    return block


class xLSTMBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(xLSTMBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )

        self.apply(
                    partial(
                        _init_weights,
                        n_layer=n_layer,
                    )
                )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])

        # Forward pass
        for layer in self.forward_blocks:
            x_forward = layer(x_forward)  
        y_forward = x_forward

        # Backward pass
        for layer in self.backward_blocks:
            x_backward = layer(x_backward)
        y_backward = torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)

class TFxLSTMBlock(nn.Module):
    """
    Temporal-Frequency xLSTM block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_xlstm (xLSTMBlock): xLSTM block for temporal dimension.
    freq_xlstm (xLSTMBlock): xLSTM block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(TFxLSTMBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Initialize xLSTM blocks
        self.time_xlstm = xLSTMBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_xlstm = xLSTMBlock(in_channels=self.hid_feature, cfg=cfg)
        
        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
    
    def forward(self, x):
        """
        Forward pass of the TFxLSTM block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency xLSTM blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_xlstm(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_xlstm(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
