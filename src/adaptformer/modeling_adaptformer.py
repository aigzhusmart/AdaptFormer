""" PyTorch AdaptFormer model."""

import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers import PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

from .configuration_adaptformer import AdaptFormerConfig


class SpatialExchange(nn.Module):

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        _, _, _, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1 = torch.zeros_like(x1, device=x1.device)
        out_x2 = torch.zeros_like(x2, device=x1.device)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class ChannelExchange(nn.Module):

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        N, c, _, _ = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1 = torch.zeros_like(x1, device=x1.device)
        out_x2 = torch.zeros_like(x2, device=x1.device)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class CascadedGroupAttention(nn.Module):
    r"""Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        kernels=[5, 5, 5, 5],
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim // (num_heads),
                        self.key_dim * 2 + self.d,
                        1,
                        1,
                        0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.key_dim * 2 + self.d),
                )
            )
            dws.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.key_dim,
                        self.key_dim,
                        kernels[i],
                        1,
                        kernels[i] // 2,
                        groups=self.key_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.key_dim),
                )
            )

        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.d * num_heads, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.act_gelu = nn.GELU()
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split(
                [self.key_dim, self.key_dim, self.d], dim=1
            )
            q = self.act_gelu(self.dws[i](q)) + q
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (q.transpose(-2, -1) @ k) * self.scale + (
                trainingab[i] if self.training else self.ab[i].to(x.device)
            )
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(nn.Module):
    r"""Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=[5, 5, 5, 5],
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, "window_size must be greater than 0"
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            dim,
            key_dim,
            num_heads,
            attn_ratio=attn_ratio,
            resolution=window_resolution,
            kernels=kernels,
        )

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert (
            H == H_ and W == W_
        ), "input feature has wrong size, expect {}, got {}".format((H, W), (H_, W_))

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (
                self.window_resolution - H % self.window_resolution
            ) % self.window_resolution
            pad_r = (
                self.window_resolution - W % self.window_resolution
            ) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            x = (
                x.view(B, nH, self.window_resolution, nW, self.window_resolution, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_resolution, self.window_resolution, C)
                .permute(0, 3, 1, 2)
            )
            x = self.attn(x)
            x = (
                x.permute(0, 2, 3, 1)
                .view(B, nH, nW, self.window_resolution, self.window_resolution, C)
                .transpose(2, 3)
                .reshape(B, pH, pW, C)
            )
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class LocalAgg(nn.Module):

    def __init__(self, channels):
        super(LocalAgg, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv2d(
            channels, channels, padding=1, kernel_size=3, groups=channels, bias=False
        )
        self.pointwise_prenorm_1 = nn.BatchNorm2d(channels)
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_prenorm_1(x)
        x = self.pointwise_conv_1(x)
        return x


class Mlp(nn.Module):

    def __init__(self, channels, mlp_ratio):
        super(Mlp, self).__init__()
        self.up_proj = nn.Conv2d(
            channels, channels * mlp_ratio, kernel_size=1, bias=False
        )
        self.down_proj = nn.Conv2d(
            channels * mlp_ratio, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class LocalMerge(nn.Module):
    def __init__(self, channels, r, heads, resolution, partial=False):
        super(LocalMerge, self).__init__()
        self.partial = partial
        self.cpe1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.local_agg = LocalAgg(channels)
        self.mlp1 = Mlp(channels, r)
        if partial:
            self.cpe2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            )
            self.attn = LocalWindowAttention(
                channels,
                16,
                heads,
                attn_ratio=r,
                resolution=resolution,
                window_resolution=7,
                kernels=[5, 5, 5, 5],
            )
            self.mlp2 = Mlp(channels, r)

    def forward(self, x):
        x = self.cpe1(x) + x
        x = self.local_agg(x) + x
        x = self.mlp1(x) + x
        if self.partial:
            x = self.cpe2(x) + x
            x = self.attn(x) + x
            x = self.mlp2(x) + x
        return x


class AdaptFormerEncoderBlock(nn.Module):
    def __init__(
        self, in_chans, embed_dim, num_head, mlp_ratio, depth, resolution, partial
    ):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=1, num_channels=embed_dim),
        )

        self.block = nn.Sequential(
            *[
                LocalMerge(
                    channels=embed_dim,
                    r=mlp_ratio,
                    heads=num_head,
                    resolution=resolution,
                    partial=partial,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.block(self.down(x))


class ChangeDetectionHaed(nn.Module):
    def __init__(self, embedding_dim, in_channels, num_classes):
        super(ChangeDetectionHaed, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim * len(in_channels),
                out_channels=embedding_dim,
                kernel_size=1,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ConvTranspose2d(embedding_dim, embedding_dim, 4, stride=2, padding=1),
        )

        self.conv1 = nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1)

        self.out = nn.Conv2d(embedding_dim, num_classes, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.conv2(F.relu(self.conv1(x))) * 0.1 + x
        return self.out(x)


class AdaptFormerDecoder(nn.Module):

    def __init__(
        self,
        config: AdaptFormerConfig,
    ):
        super(AdaptFormerDecoder, self).__init__()

        self.in_channels = config.embed_dims
        self.embedding_dim = config.embed_dims[-1]

        self.linear_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange("n c ... -> n (...) c"),
                    nn.Linear(in_dim, self.embedding_dim),
                )
                for in_dim in self.in_channels
            ]
        )

        self.diff_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2 * self.embedding_dim, self.embedding_dim, 3, 1, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.embedding_dim),
                    nn.Conv2d(self.embedding_dim, self.embedding_dim, 3, 1, 1),
                    nn.ReLU(),
                )
                for _ in range(3)
            ]
        )

        self.prediction_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.embedding_dim, config.num_classes, 3, 1, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(config.num_classes),
                    nn.Conv2d(config.num_classes, config.num_classes, 3, 1, 1),
                )
                for _ in range(3)
            ]
        )

        self.head = ChangeDetectionHaed(
            self.embedding_dim, self.in_channels, config.num_classes
        )

    def forward(self, pixel_valuesA, pixel_valuesB):
        N, _, H, W = pixel_valuesA[0].shape

        # c3
        pixel_values_c3 = torch.cat([pixel_valuesA[2], pixel_valuesB[2]], dim=0)

        _c3_1, _c3_2 = torch.chunk(
            self.linear_emb_layers[2](pixel_values_c3).permute(0, 2, 1), 2
        )
        _c3_1 = _c3_1.reshape(N, -1, pixel_values_c3.shape[2], pixel_values_c3.shape[3])
        _c3_2 = _c3_2.reshape(N, -1, pixel_values_c3.shape[2], pixel_values_c3.shape[3])

        _c3 = self.diff_layers[2](torch.cat((_c3_1, _c3_2), dim=1))

        p_c3 = self.prediction_layers[2](_c3)
        _c3_up = F.interpolate(_c3, (H, W), mode="bilinear", align_corners=False)

        # c2
        pixel_values_c2 = torch.cat([pixel_valuesA[1], pixel_valuesB[1]], dim=0)
        _c2_1, _c2_2 = torch.chunk(
            self.linear_emb_layers[1](pixel_values_c2).permute(0, 2, 1), 2
        )
        _c2_1 = _c2_1.reshape(N, -1, pixel_values_c2.shape[2], pixel_values_c2.shape[3])
        _c2_2 = _c2_2.reshape(N, -1, pixel_values_c2.shape[2], pixel_values_c2.shape[3])
        _c2 = self.diff_layers[1](torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(
            _c3, scale_factor=2, mode="bilinear"
        )
        p_c2 = self.prediction_layers[1](_c2)
        _c2_up = F.interpolate(_c2, (H, W), mode="bilinear", align_corners=False)

        # c1
        pixel_values_c1 = torch.cat([pixel_valuesA[0], pixel_valuesB[0]], dim=0)
        _c1_1, _c1_2 = torch.chunk(
            self.linear_emb_layers[0](pixel_values_c1).permute(0, 2, 1), 2
        )
        _c1_1 = _c1_1.reshape(N, -1, pixel_values_c1.shape[2], pixel_values_c1.shape[3])
        _c1_2 = _c1_2.reshape(N, -1, pixel_values_c1.shape[2], pixel_values_c1.shape[3])
        _c1 = self.diff_layers[0](torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(
            _c2, scale_factor=2, mode="bilinear"
        )
        p_c1 = self.prediction_layers[0](_c1)

        cp = self.head(torch.cat((_c3_up, _c2_up, _c1), dim=1))

        return [p_c3, p_c2, p_c1, cp]


class AdaptFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AdaptFormerConfig
    base_model_prefix = "adaptformer"

    def _init_weights(self, m):
        """Initialize the weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            import math

            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class AdaptFormerForChangeDetection(AdaptFormerPreTrainedModel):
    """
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`AdaptFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(
        self,
        config: AdaptFormerConfig,
    ):
        super().__init__(config)
        self.config = config
        self.block1 = AdaptFormerEncoderBlock(
            in_chans=config.num_channels,
            embed_dim=config.embed_dims[0],
            num_head=config.num_heads[0],
            mlp_ratio=config.mlp_ratios[0],
            depth=config.depths[0],
            resolution=config.embed_dims[2] // 2,
            partial=False,
        )
        self.block2 = AdaptFormerEncoderBlock(
            in_chans=config.embed_dims[0],
            embed_dim=config.embed_dims[1],
            num_head=config.num_heads[1],
            mlp_ratio=config.mlp_ratios[1],
            depth=config.depths[1],
            resolution=config.embed_dims[1] // 2,
            partial=False,
        )
        self.block3 = AdaptFormerEncoderBlock(
            in_chans=config.embed_dims[1],
            embed_dim=config.embed_dims[2],
            num_head=config.num_heads[2],
            mlp_ratio=config.mlp_ratios[2],
            depth=config.depths[2],
            resolution=config.embed_dims[0] // 2,
            partial=True,
        )
        self.spatialex = SpatialExchange()
        self.channelex = ChannelExchange()

        self.decoder = AdaptFormerDecoder(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_valuesA: torch.Tensor,
        pixel_valuesB: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("deepang/adaptformer-LEVIR-CD")
        >>> model = AutoModel.from_pretrained("deepang/adaptformer-LEVIR-CD")

        >>> image_A = Image.open(requests.get('https://raw.githubusercontent.com/aigzhusmart/AdaptFormer/main/figures/test_2_1_A.png', stream=True).raw)
        >>> image_B = Image.open(requests.get('https://raw.githubusercontent.com/aigzhusmart/AdaptFormer/main/figures/test_2_1_B.png', stream=True).raw)
        >>> label = Image.open(requests.get('https://raw.githubusercontent.com/aigzhusmart/AdaptFormer/main/figures/test_2_1_label.png', stream=True).raw)

        >>> with torch.no_grad():
        >>>     inputs = preprocessor(images=(image_A, image_B), return_tensors="pt")
        >>>     outputs = adaptfromer_model(**inputs)
        >>>     logits = outputs.logits.cpu()
        >>>     pred = logits.argmax(dim=1)[0]
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        x1_1, x2_1 = torch.chunk(
            self.block1(torch.cat((pixel_valuesA, pixel_valuesB), dim=0)), 2
        )

        x1_2, x2_2 = torch.chunk(
            self.block2(torch.cat(self.spatialex(x1_1, x2_1), dim=0)), 2
        )

        x1_3, x2_3 = torch.chunk(
            self.block3(torch.cat(self.channelex(x1_2, x2_2), dim=0)), 2
        )

        hidden_states = self.decoder([x1_1, x1_2, x1_3], [x2_1, x2_2, x2_3])

        loss = None
        if labels is not None:
            loss = 0
            for i, hidden_state in enumerate(hidden_states):
                upsampled_logits = F.interpolate(
                    hidden_state,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss += (
                    F.cross_entropy(
                        upsampled_logits,
                        labels.long(),
                        ignore_index=self.config.semantic_loss_ignore_index,
                    )
                    * self.config.semantic_loss_weight[i]
                )

        if not return_dict:
            if output_hidden_states:
                output = (hidden_states[-1], hidden_states)
            else:
                output = (hidden_states[-1],)
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=hidden_states[-1],
            hidden_states=hidden_states if output_hidden_states else None,
        )
