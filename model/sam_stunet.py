# %%writefile /kaggle/working/mdm/model/sam_stunet.py
from abc import abstractmethod
from sentence_transformers import SentenceTransformer, models
import math 
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


from diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from model.qna import FusedQnA1d


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, up_size=None):
        x = x.type(torch.cuda.FloatTensor)
        emb = emb.type(torch.cuda.FloatTensor)
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x, emb_out = layer(x, emb, up_size)
            else:
                x = layer(x)
        return x, None
    
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding_mode='zeros', padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode)

    def forward(self, x, up_size):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
#             x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.interpolate(x, size=up_size, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding_mode='zeros', padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, padding_mode=padding_mode
            )
        else:
            assert self.channels == self.out_channels
            kernel_size = 7
            self.op = avg_pool_nd(dims, kernel_size=kernel_size, stride=stride,  padding=(kernel_size - 1) // 2)
            # self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, up_size=None):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=1,
        use_checkpoint=False,
        up=False,
        down=False,
        padding_mode='zeros',
        padding=1,
        in_cbam=False
    ):
        super().__init__()
        print(f"Dropout in Res : {dropout}")

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_cbam = in_cbam

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode),
        )
        

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Upsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        elif down:
            self.h_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
            self.x_upd = Downsample(channels, False, dims, padding_mode=padding_mode, padding=padding)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.in_cbam:
            self.emb_layers = nn.Identity()
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
            
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode)
            ),
        )

        if self.out_channels == channels or in_cbam:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=padding, padding_mode=padding_mode
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, up_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, up_size), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, up_size):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h,up_size)
            x = self.x_upd(x,up_size)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if self.in_cbam:
            return self.skip_connection(x) + h
        else:
            return self.skip_connection(x) + h, emb_out

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class SAM(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=False, qk_scale=None, 
                       attn_drop=0., dropout=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        print(f"Dropout in SAM : {dropout}")

        self.act = nn.GELU()
        self.proj_out = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.proj_drop = nn.Dropout(dropout)

        self.split_groups=self.dim//ca_num_heads
        

        self.gru_s = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, \
                          bias=True, batch_first=True, dropout=0.0, \
                          bidirectional=True)


        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        for i in range(self.ca_num_heads):
            local_conv = nn.Conv1d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=i + 1, stride=1, groups=dim//self.ca_num_heads)
            setattr(self, f"f_local_conv_{i + 1}", local_conv)

        for i in range(self.ca_num_heads):
            local_conv = nn.Conv1d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=i + 1, stride=1, groups=dim//self.ca_num_heads)
            setattr(self, f"b_local_conv_{i + 1}", local_conv)
        
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups),
            nn.BatchNorm1d(dim*expand_ratio),
            nn.SiLU(),
            nn.Conv1d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.apply(self._init_weights)

        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    
    def forward(self, x):
        orin_x = x
        x = self.norm(x.permute(0, 2, 1))
        B, N, C = x.shape
        v = self.v(x)
        ss = self.gru_s(x)[0]
        fs, bs = torch.chunk(ss, 2, dim=-1)
        fs = fs.reshape(B, N, self.ca_num_heads, C//self.ca_num_heads).permute(2, 0, 3, 1)
        bs = bs.reshape(B, N, self.ca_num_heads, C//self.ca_num_heads).permute(2, 0, 3, 1)
        
        for i in range(self.ca_num_heads):
            f_local_conv = getattr(self, f"f_local_conv_{i + 1}")
            b_local_conv = getattr(self, f"b_local_conv_{i + 1}")
            fs_i = fs[i]
            bs_i = bs[i]
            fs_i = f_local_conv(fs_i).reshape(B, self.split_groups, -1, N) # + g_i.reshape(B, self.split_groups, -1, N)
            bs_i = b_local_conv(bs_i).reshape(B, self.split_groups, -1, N) # + g_i.reshape(B, self.split_groups, -1, N)

            s_i = fs_i + bs_i
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2)
                
        s_out = s_out.reshape(B, C, N)
        s_out = self.proj(s_out)
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        
        out = s_out * v
        
        out = self.proj_out(out.permute(0, 2, 1))
        out = self.proj_drop(out)
        return orin_x + out

    
class SAM_UNetModel(nn.Module):
    def __init__(
        self,
        cond_mask_prob,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0.5,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        padding_mode='zeros',
        padding=1,
        cond_mode="text",
    ):
        super().__init__()
        
        self.cond_mode = cond_mode

        self.cond_mask_prob = cond_mask_prob
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.sub_sample_mult = np.power(2, len(self.channel_mult))
        self.dims = dims
        self.padding_mode = padding_mode
        self.padding = padding
        

        num_heads_upsample=num_heads


        self.text_dropout = nn.Dropout(0.0)

        time_embed_dim = 1024

        print(f"TEXT-EMB-DIM : {time_embed_dim}")

        assert time_embed_dim == 1024, f"Text dimension is invalid : {time_embed_dim}"




        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0]) * model_channels
        

        in_list = [conv_nd(dims, in_channels, ch, 3, padding=padding, padding_mode=self.padding_mode)]
        
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(*in_list)]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        max_len = 500
        save_len = [max_len, max_len]
        
        out_ch = ch // 2
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            padding_mode=padding_mode,
                            padding=padding,
                            down=True,
                        ),
                ]
                
                print(f"ENCODER : {ch} -> {out_ch}")
                
                max_len = math.ceil(max_len / 2)
                save_len.append(max_len)


                layers.append(SAM(dim=out_ch, ca_num_heads=num_heads_upsample))
                
                ch = out_ch
                out_ch = ch // 2

                self.input_blocks.append(TimestepEmbedSequential(*layers)).to('cuda')
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers = [
                    ResBlock(
                        ich + ch,
                        time_embed_dim,
                        dropout,
                        out_channels=ch*2,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        padding_mode=padding_mode,
                        padding=padding,
                        up=True,
                    ),
                ]
                print(f"DECODER : {ich + ch} -> {ch*2}")
                
                max_len = save_len.pop()
                

                layers.append(SAM(dim=ch*2, ca_num_heads=num_heads_upsample))
                
                
                ch = ch * 2

                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            padding_mode=padding_mode,
                            padding=padding,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, padding_mode=padding_mode, padding=padding)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=padding, padding_mode=padding_mode)),
        )

        print('EMBED TEXT')
        print('Loading clip-ViT-B-32-multilingual-v1...')

        clip_version = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
        self.clip_model = self.load_and_freeze_clip(clip_version)
        self.clip_embed_text = nn.Linear(768, time_embed_dim)
        
        print(f"Probability for masking condition : {self.cond_mask_prob}")
        

    def encode_text(self, raw_text):
        embeddings = torch.tensor(self.clip_model.encode(raw_text)).float().to('cuda')
        return embeddings


    def emb_text(self, enc_clip, force_mask):
        emb_clip = self.clip_embed_text(self.mask_cond(enc_clip, force_mask))
        return self.text_dropout(emb_clip)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not (name.startswith('clip_model.') or name.startswith('bert_model.'))]

    def load_and_freeze_clip(self, clip_version):
        word_embedding_model = models.Transformer(clip_version)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        clip_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) # Actually this line is unnecessary since clip by default already on float16

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model.to('cuda')

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def mask_cond(self, cond, force_mask):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = th.bernoulli(th.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """ 

        x = x.squeeze(dim=2) # print(x.shape)  # ([2, 150, 500])
#         x = x.reshape(x.size(0), -1, x.size(2))

        self.n_samples, self.n_feats, self.n_frames = x.shape

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if 'text_embed' in y.keys():  # caching option
            enc_text_clip = y['text_embed']
        else:
            enc_text_clip = self.encode_text(y['text'])

        force_mask = y.get('uncond', False)
        emb += self.emb_text(enc_text_clip, force_mask=force_mask)

        x = x.reshape(self.n_samples, -1, self.n_frames)


        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)  # if y is None else th.cat([x, y], dim=1).type(self.dtype)

        for module in self.input_blocks:
            h, _ = module(h, emb)
            hs.append(h)
            
        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h = hs.pop()
            else:
                h = th.cat([h, hs.pop()], dim=1)
            if len(hs) > 0:
                h, _ = module(h, emb, up_size=hs[-1].shape[-1])
            else:
                h, _ = module(h, emb, up_size=self.n_frames)

        h = h.type(x.dtype)

        _out = self.out(h)


        _out = _out.reshape(self.n_samples, self.n_feats, 1, self.n_frames)
        return _out


