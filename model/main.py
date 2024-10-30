import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as func
from einops import rearrange
from einops.layers.torch import Rearrange
from .layers import Unit1D
from .boundary_pooling_op import BoundaryMaxPooling
from .i3d_backbone import InceptionI3d


class I3DBackBone(nn.Module):
    def __init__(self, in_channels, final_endpoint='Mixed_5c', name='inception_i3d',
                 freeze_bn=True, freeze_bn_affine=True):
        super(I3DBackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint=final_endpoint,
                                   name=name,
                                   in_channels=in_channels)
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def load_pretrained_weight(self, model_path):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def train(self, mode=True):
        super(I3DBackBone, self).train(mode)
        if self._freeze_bn and mode:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)


# class MyGELU(nn.Module):
#     def __int__(self):
#         super().__int__()
#
#     def forward(self, x):
#         return func.gelu(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, in_channels, conv_channels, depth,
                 heads=4, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4):
        super().__init__()
        dim = conv_channels
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding[:, :, :n]
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        x = self.temporal_transformer(x)
        x = rearrange(x, 'b t d -> b d t')
        return x


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input_tensor):
        return torch.exp(input_tensor * self.scale)


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments):
        # return None
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature


class CoarsePyramid(nn.Module):
    def __init__(self, feat_pro, frame_num, num_classes, conv_channels, layer_num, depth, feat_t):
        super().__init__()
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        out_channels = conv_channels
        self.frame_num = frame_num
        self.num_classes = num_classes
        for item in feat_pro:
            self.pyramids.append(nn.Sequential(
                ViViT(
                    conv_channels=conv_channels,
                    num_frames=item['num_frames'],
                    in_channels=item['in_channels'],
                    image_size=item['image_size'],
                    patch_size=item['patch_size'],
                    depth=depth
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        for i in range(2, layer_num):
            self.pyramids.append(nn.Sequential(
                Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        self.conf_proposal_branch = ProposalBranch(out_channels, 512)

        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2

    def forward(self, feat_dict, ssl=False):
        x2 = feat_dict['Mixed_5c']
        x1 = feat_dict['Mixed_4f']
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        batch_num = x1.size(0)
        trip = []
        x = None
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(x1)
            elif i == 1:
                x = conv(x2)
                x0 = pyramid_feats[-1]
                y = func.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)
            pyramid_feats.append(x)
        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = func.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        start_loc_prop = None
        end_loc_prop = None
        start_conf_prop = None
        end_conf_prop = None

        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)

            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            confs.append(
                self.conf_head(conf_feat).view(batch_num, self.num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )
            t = feat.size(2)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                priors = self.priors[i].expand(batch_num, t, 1).to(feat.device)
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)

            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(loc_feat, frame_level_feat,
                                                                     segments, frame_segments)
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(conf_feat, frame_level_feat,
                                                                        segments, frame_segments)
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                if ssl:
                    return trip
            prop_locs.append(self.prop_loc_head(loc_prop_feat).view(batch_num, 2, -1)
                             .permute(0, 2, 1).contiguous())
            prop_confs.append(self.prop_conf_head(conf_prop_feat).view(batch_num, self.num_classes, -1)
                              .permute(0, 2, 1).contiguous())
            centers.append(
                self.center_head(loc_prop_feat).view(batch_num, 1, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in confs], 1)
        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, prop_loc, prop_conf, center, priors, start, end, \
            start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop


def calculate_fan_in_and_fan_out(tensor):  # the same as nn.init._calculate_fan_in_and_fan_out
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def no_grad_uniform_(tensor, a, b):  # the same as nn.init._no_grad_uniform_
    with torch.no_grad():
        return tensor.uniform_(a, b)


class MyModel(nn.Module):  # the core module of TeST
    def __init__(self, feat_pro, in_channels, num_classes, conv_channels, layer_num, depth, feat_t,
                 backbone_model, frame_num, training=True):
        super().__init__()

        self.coarse_pyramid_detection = CoarsePyramid(feat_pro, frame_num, num_classes,
                                                      conv_channels, layer_num, depth, feat_t)
        self.reset_params()

        self.backbone = I3DBackBone(in_channels=in_channels)
        self.boundary_max_pooling = BoundaryMaxPooling()
        self._training = training

        if self._training:
            self.backbone.load_pretrained_weight(backbone_model)
        self.scales = [1, 4, 4]

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, proposals=None, ssl=False):
        feat_dict = self.backbone(x)
        if ssl:
            top_feat = self.coarse_pyramid_detection(feat_dict, ssl)
            decoded_segments = proposals[0].unsqueeze(0)
            plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
            in_plen = torch.clamp(plen / 4.0, min=1.0)
            out_plen = torch.clamp(plen / 10.0, min=1.0)
            frame_segments = torch.cat([
                torch.round(decoded_segments[:, :, :1] - out_plen),
                torch.round(decoded_segments[:, :, :1] + in_plen),
                torch.round(decoded_segments[:, :, 1:] - in_plen),
                torch.round(decoded_segments[:, :, 1:] + out_plen)
            ], dim=-1)
            anchor, positive, negative = [], [], []
            for i in range(3):
                bound_feat = self.boundary_max_pooling(top_feat[i], frame_segments / self.scales[i])
                # for triplet loss
                ndim = bound_feat.size(1) // 2
                anchor.append(bound_feat[:, ndim:, 0])
                positive.append(bound_feat[:, :ndim, 1])
                negative.append(bound_feat[:, :ndim, 2])

            return anchor, positive, negative
        else:
            loc, conf, prop_loc, prop_conf, center, priors, start, end, \
                start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop = self.coarse_pyramid_detection(feat_dict)
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                'prop_loc': prop_loc,
                'prop_conf': prop_conf,
                'center': center,
                'start': start,
                'end': end,
                'start_loc_prop': start_loc_prop,
                'end_loc_prop': end_loc_prop,
                'start_conf_prop': start_conf_prop,
                'end_conf_prop': end_conf_prop
            }
