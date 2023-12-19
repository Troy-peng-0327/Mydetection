import torch
import torch.nn as nn

EPSILON = 1e-10


class Fusion_ADD(nn.Module):
    def forward(self, feat_t, feat_v):
        fused_feat = feat_t + feat_v
        return fused_feat


class Fusion_AVG(nn.Module):
    def forward(self, feat_t, feat_v):
        fused_feat = (feat_t + feat_v) / 2
        return fused_feat


class Fusion_MAX(nn.Module):
    def forward(self, feat_t, feat_v):
        fused_feat = torch.max(feat_t, feat_v)


class Fusion_CAT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
    
    def forward(self, feat_t, feat_v):
        fused_feat = self.conv1(torch.cat((feat_t, feat_v), dim=1))
        return fused_feat
    

class Fusion_GATED(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1_t = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feat_t, feat_v):
        t_gate = self.sigmoid(self.conv1_t(feat_t))
        v_gate = self.sigmoid(self.conv1_v(feat_v))
        return t_gate * feat_t + v_gate * feat_v
    

class Fusion_SPA(nn.Module):
    def forward(self, feat_t, feat_v):
        shape = feat_t.size()

        # calculate spatial attention
        spatial_t = spatial_attention(feat_t, spatial_type='mean')
        spatial_v = spatial_attention(feat_v, spatial_type='mean')

        # get weight map, soft-max
        spatial_weight_t = torch.exp(spatial_t) / (torch.exp(spatial_t) + torch.exp(spatial_v) + EPSILON)
        spatial_weight_v = torch.exp(spatial_v) / (torch.exp(spatial_t) + torch.exp(spatial_v) + EPSILON)
        spatial_weight_t = spatial_weight_t.repeat(1, shape[1], 1, 1)
        spatial_weight_v = spatial_weight_v.repeat(1, shape[1], 1, 1)
        return spatial_weight_t * feat_t + spatial_weight_v * feat_v

# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_strategy, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_cat = Fusion_CAT(in_channels=in_channels)
        self.fusion_spa = Fusion_SPA()
        self.fusion_gated = Fusion_GATED(in_channels=in_channels)

    def forward(self, feat_t, feat_v, fs_type):
        self.fs_type = fs_type
        if self.fs_type == 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type == 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type == 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type == 'cat':
            fusion_operation = self.fusion_cat
        elif self.fs_type == 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type == 'gated':
            fusion_operation = self.fusion_gated
        else:
            raise ValueError(
                f'There is no fusion_operation {self.fs_type}.')
        
        if isinstance(feat_t, tuple) or isinstance(feat_t, list):
            fused_feat = []
            for i in range(len(feat_t)):
                fused_feat.append(fusion_operation(feat_t[i], feat_v[i]))
        else:
            fused_feat = fusion_operation(feat_t, feat_v)

        return fused_feat
