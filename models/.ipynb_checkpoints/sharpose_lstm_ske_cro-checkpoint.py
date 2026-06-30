from functools import partial
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmpose.utils import get_root_logger
from mmpose.models.builder import BACKBONES
from mmpose.datasets.builder import PIPELINES 

import math
import random


@PIPELINES.register_module()
class CustomRandomErasing:
    def __init__(self, probability=0.3, area_ratio_range=(0.02, 0.1), min_aspect=0.3, max_aspect=1.0):
        self.prob = probability
        self.area_ratio_range = area_ratio_range
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        
        img = results['img'] 
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            h, w = img.shape
            c = 1
        
        area = h * w
        
        for _ in range(10):
            target_area = np.random.uniform(*self.area_ratio_range) * area
            aspect_ratio = np.random.uniform(self.min_aspect, self.max_aspect)
            eh = int(round(np.sqrt(target_area * aspect_ratio)))
            ew = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if eh < h and ew < w:
                x = np.random.randint(0, w - ew + 1)
                y = np.random.randint(0, h - eh + 1)
                if c == 1:
                    img[y:y+eh, x:x+ew] = np.random.uniform(0, 1, (eh, ew))
                else:
                    img[y:y+eh, x:x+ew, :] = np.random.uniform(0, 1, (eh, ew, c))
                break
                
        results['img'] = img
        return results
    
@PIPELINES.register_module()
class RandomKeypointMask:
    def __init__(self, prob=0.3, min_mask=1, max_mask=3):
        self.prob = prob
        self.min_mask = min_mask
        self.max_mask = max_mask

    def __call__(self, results):
        if random.random() > self.prob:
            return results       
        num_joints = results['joints_3d'].shape[0]
        mask_num = random.randint(self.min_mask, min(self.max_mask, num_joints))
        mask_indices = random.sample(range(num_joints), mask_num)
        for idx in mask_indices:
            results['joints_3d_visible'][idx] = 0
        return results

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def reorder_input(x, new_order):
    """
    :param x: 输入数据，形状为 (batch_size, seq_len, input_dim)
    :param new_order: 用于改变关键点顺序的索引列表，形状为 (seq_len,)
    :return: 重新排序后的输入数据，形状为 (batch_size, seq_len, input_dim)
    """
    return x[:, new_order, :]
def recover_output(output, reverse_order):
    """
    :param output: LSTM的输出数据，形状为 (batch_size, seq_len, output_dim)
    :param reverse_order: 用于恢复顺序的反向排序索引
    :return: 恢复顺序后的输出数据，形状为 (batch_size, seq_len, output_dim)
    """
    return output[:, reverse_order, :] 

# 在细粒度（fine stage）阶段中，如何根据粗粒度（coarse stage）阶段的
# 索引来获取相应的索引。这个函数通常用于处理图像分块（patches）或多尺度特征映射，
def get_index(idx, patch_shape_src, patch_shape_des):
    '''
    get index of fine stage corresponding to coarse stage 
    '''
    h1, w1 = patch_shape_src
    h2, w2 = patch_shape_des
    hs = h2 // h1
    ws = w2 // w1            
    
    j = idx % w1
    i = torch.div(idx, w1, rounding_mode='floor')
    
    idx = i * hs * w2 + j * ws
    
    idxs = []
    for i in range(hs):
        for j in range(ws):
            idxs.append(idx + i * w2 + j)
    
    return torch.cat(idxs, dim=1)

# 是对两个元组（tp1 和 tp2）中的对应元素进行整数除法操作，
# 并返回一个新的元组。这个函数通常用于处理与形状、尺寸或其他相关数据的逐元素计算。
def tuple_div(tp1, tp2):
    return tuple(i // j for i, j in zip(tp1, tp2))

# 用于将输入图像划分为多个补丁（patches），并将这些补丁嵌入到一个更高维度的表示空间中。输入图像的大小，每个
# 补丁的大小，输入图像的通道数，嵌入后的维度输入为（B，C，H，W），通过卷积层将输入转换为补丁表示
# 使用 flatten(2) 将输出从形状 (B, embed_dim, H', W') 展平为 (B, embed_dim, H' * W')，这里的 H' 和 W' 是经过卷积后得到的高度和宽度。
# 使用 transpose(1, 2) 交换维度，使输出的形状变为 (B, H' * W', embed_dim)，每个补丁的嵌入表示都是这个形状。
class MultiResoPatchEmbed(nn.Module):
    def __init__(self, img_sizes=[112, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_sizes = [*map(to_2tuple, img_sizes)]
        patch_size = to_2tuple(patch_size)
        self.patch_shapes = [*map(partial(tuple_div, tp2=patch_size), img_sizes)]
        self.patch_size = patch_size
        self.num_patches = [H * W for H, W in self.patch_shapes]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# 用于实现多头注意力机制，dim输入特征的维度，多头注意力的数量，是否在QKV的计算过程中使用偏置，Q，K的缩放因子
class Attention(nn.Module):
    """
        return the attention map
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads   
        self.scale = qk_scale or head_dim ** -0.5  

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) 

    def forward(self, x): 
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)  
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)  
        x = self.proj_drop(x)
        return x, attn   

# 构建transformer模块，dim输入特征维度，注意力头的数量，定义 MLP 隐藏层的维度与输入维度的比率（默认为 4.），即隐藏层维度 = dim * mlp_ratio。
# 计算kqv中的偏置，Q和K的缩放因子，dropout的概率，注意力权重的dropout的概率，随机深度的dropout概率，激活函数，归一化层
class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x): 
        x2, atten = self.attn(self.norm1(x))
        x = x + self.drop_path(x2) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x, atten

# 神经网络模块，用于进行质量预测,embed输入特征维度，drop比率，激活函数，归一化层，输出是否使用sigmoid激活，输出是否为绝对质量预测（通常为分类问题）
class QualityPredictor(nn.Module):
    def __init__(self, embed=768, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 sigmoid=False, qp_abs=False) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(  
            nn.Linear(embed, embed),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(embed, embed),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(embed, 2 if qp_abs else 1),  
            nn.Softmax(dim=-1) if qp_abs else nn.Sigmoid() if sigmoid else act_layer(),
        )
        self.norm = norm_layer(embed)
        
    def forward(self, x: torch.Tensor):
        x = x.mean(dim=1)  
        x = self.norm(x)
        x = self.mlp(x)
        return x


import torch
import torch.nn as nn


class BiLSTMKeypointModel(nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        """
        :param input_dim: 输入的特征维度，假设每个关键点的token是一个向量，input_dim是该向量的维度
        :param hidden_dim: LSTM的隐藏层维度
        :param output_dim: 输出的特征维度，通常与input_dim相同，因为我们要输出关键点的坐标
        :param num_layers: LSTM的层数
        :param dropout: Dropout层的丢弃概率，防止过拟合
        """
        super(BiLSTMKeypointModel, self).__init__()

        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  

    def forward(self, x):
        """
        :param x: 输入，形状为 (batch_size, seq_len, input_dim)，batch_size为批次大小，seq_len为序列长度
        :return: 输出，形状为 (batch_size, seq_len, output_dim)，每个时间步输出一个关键点token
        """
        
        lstm_out, _ = self.lstm(x)

        output = self.fc(lstm_out)

        return output


class KeypointsToSkeletonGCN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_keypoints=17, num_skeletons=8): 
        super(KeypointsToSkeletonGCN, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_skeletons = num_skeletons

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, keypoints, edge_index):
        """
        keypoints: (64, 17, 768) 输入关键点特征
        edge_index: (2, num_edges) 图的边列表
        """
        batch_size = keypoints.size(0)

        keypoints = keypoints.reshape(-1, keypoints.size(-1))  # (64 * 17, 768)
        x = self.gcn1(keypoints, edge_index)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.gcn2(x, edge_index)
        x = self.norm2(x)
        x = F.gelu(x)

        x = x.reshape(batch_size, self.num_keypoints, -1)  # (64, 17, hidden_dim)


        skeletons = []
        for i in range(self.num_skeletons):

            idx1, idx2 = edge_index[:, i] 

            point1 = x[:, idx1, :]  # (64, hidden_dim)
            point2 = x[:, idx2, :]  # (64, hidden_dim)

            combined = torch.cat([point1, point2], dim=-1)  # (64, 2 * hidden_dim)

            skeleton_feature = self.fusion_layer(combined)  # (64, input_dim)

            skeletons.append(skeleton_feature)

        skeletons = torch.stack(skeletons, dim=1)  # (64, 8, input_dim)

        return skeletons

class JointRelationModule(nn.Module): 
    def __init__(self, keypoints=17):
        super().__init__()
        self.kpts_num = keypoints 

        self.kpts_conv_k = nn.Conv2d(self.kpts_num, self.kpts_num, kernel_size=(1, 1), stride=1, padding=0)
        self.kpts_conv_q = nn.Conv2d(self.kpts_num, self.kpts_num, kernel_size=(1, 1), stride=1, padding=0)
        self.kpts_conv_v = nn.Conv2d(self.kpts_num, self.kpts_num, kernel_size=(1, 1), stride=1, padding=0)

        self.norm_fact = self.kpts_num ** 0.5  
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, kpt_feat):
        # 假设 kpt_feat 的形状为 [B, C, H, W]
        B, C, H, W = kpt_feat.size()  # 获取输入的特征大小
        residual_kpt_feat = kpt_feat  # 保存输入特征

        kpts_feat_k = self.kpts_conv_k(kpt_feat).view(B, self.kpts_num, -1).contiguous()  # [B, kpts, dim]
        kpts_feat_q = self.kpts_conv_q(kpt_feat).view(B, self.kpts_num, -1).contiguous()  # [B, kpts, dim]
        kpts_feat_v = self.kpts_conv_v(kpt_feat).view(B, self.kpts_num, -1).contiguous()  # [B, kpts, dim]

        kpt_feats = []
        for i in range(B):
            kpt_relation = torch.matmul(kpts_feat_q[i:i + 1], kpts_feat_k[i:i + 1].permute(0, 2,
                                                                                           1).contiguous()) / self.norm_fact  # [1, kpts, kpts]
            kpt_relation = F.softmax(kpt_relation, dim=-1) 


            kpt_relation_feat = torch.matmul(kpt_relation, kpts_feat_v[i:i + 1]).view(1, self.kpts_num, H,
                                                                                      W).contiguous()  # [1, kpts, h, w]
            kpt_feats.append(kpt_relation_feat)


        kpt_feats = torch.cat(kpt_feats, dim=0)  # [B, kpts, h, w]
        kpt_feats = self.gamma * kpt_feats + residual_kpt_feat

        return F.relu(kpt_feats)

# 交叉注意力层
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
            )    
        self.proj_drop = nn.Dropout(proj_drop)

        self.final_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.final_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(proj_drop)
        )
        self.linear_k = nn.Linear(64, 768)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv1.weight)
        nn.init.xavier_uniform_(self.qkv2.weight)
        nn.init.xavier_uniform_(self.final_q.weight)
        nn.init.xavier_uniform_(self.final_kv.weight)
        nn.init.xavier_uniform_(self.proj.weight)

        if self.qkv1.bias is not None:
            nn.init.constant_(self.qkv1.bias, 0)
            nn.init.constant_(self.qkv2.bias, 0)
            nn.init.constant_(self.final_q.bias, 0)
            nn.init.constant_(self.final_kv.bias, 0)
            nn.init.constant_(self.proj.bias, 0)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x1, x2): 
        B, N, C = x1.shape

        qkv1 = self.qkv1(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.qkv2(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)

        q_final = self.final_q(out1)

        kv_final = self.final_kv(out2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v_final = self.final_q(out2)

        k_final = kv_final[0]

        k_final = k_final.reshape(B, N, 768)  

        attn_final = (q_final @ k_final.transpose(-2, -1)) * self.scale 
        attn_final = attn_final.softmax(dim=-1)
        attn_final = self.attn_drop(attn_final)

        fused_out = (attn_final @ v_final).transpose(1, 2).reshape(B, N, C)
        fused_out = self.proj(fused_out)
        gate = self.gate(x1)
        fused_out = x1 + gate * self.proj_drop(fused_out)
        out = fused_out + self.mlp(fused_out)

        return out



@BACKBONES.register_module()   
class SHaRPoseLstmSkeCro(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    输入图像尺寸，图像划分为补丁的大小，输入的通道数，分类任务的类别数，嵌入维度，网络的深度（transformer层的数量），注意力头的数量，MLP维度的比例
    注意力机制和dropout的超参数，归一化层，关键点数量，质量预测相关的参数，用于选择重要补丁的比例，是否替换OKS实例
    
    """
    def __init__(self, img_sizes=[112, 224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, num_keypoints=17,
                 qp_threshold=0.95, qp_start_epoch=0, qp_sigmoid=False, qp_abs=False,
                 alpha=0.5, replace_oks=False):
        super().__init__()    
        self.informative_selection = True  
        self.alpha = alpha  
        self.beta = 0.99  
        self.ske_num = 8
        self.target_index = [*range(depth // 4, depth)]   
        
        self.img_sizes = [*map(to_2tuple, img_sizes)]  

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.num_keypoints = num_keypoints
        self.keypoint_tokens = nn.Parameter(torch.zeros(1, num_keypoints, embed_dim))
        self.skeleton_tokens = nn.Parameter(torch.zeros(1, self.ske_num, embed_dim))
        
        self.patch_embed = MultiResoPatchEmbed(
            img_sizes=img_sizes, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed_list = nn.ParameterList([  
            nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            for num_patches in self.patch_embed.num_patches
        ])
        self.pos_drop = nn.Dropout(p=drop_rate)  

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([    
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  


        self.reuse_block = nn.Sequential(
            norm_layer(embed_dim),
            Mlp(in_features=embed_dim, hidden_features=mlp_ratio*embed_dim,out_features=embed_dim,act_layer=nn.GELU,drop=drop_rate)
        ) 
    
        self.quality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  
        self.quality_predictor = QualityPredictor(embed_dim, drop=drop_rate, sigmoid=qp_sigmoid, qp_abs=qp_abs)
        self.qp_threshold = qp_threshold
        self.qp_start_epoch = qp_start_epoch
        self.qp_abs = qp_abs  
        self.train_epoch = None
        self.replace_oks = replace_oks  
        self.bilstm = BiLSTMKeypointModel(embed_dim,256,embed_dim)
        self.keypoint_ske = KeypointsToSkeletonGCN()
        self.joint_relation = JointRelationModule(keypoints=num_keypoints)  
        self.cross_attention = CrossAttentionBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.local_conv = nn.Sequential(
            nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, padding=1, groups=num_keypoints, bias=False),
            nn.GroupNorm(1, num_keypoints),   
            nn.GELU()
        )

    def init_weights(self, pretrained=None):   
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)   
                if isinstance(m, nn.Linear) and m.bias is not None:     
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)  
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:  
                        nn.init.xavier_uniform_(param)  
                    elif 'weight_hh' in name:  
                        nn.init.orthogonal_(param)  
                    elif 'bias' in name:  
                        nn.init.constant_(param, 0)  

        if isinstance(pretrained, str):  
            self.apply(_init_weights)  
            logger = get_root_logger()    
            logger.info(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)  
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, img: torch.Tensor):     
        results = []
        ske = []
        global_attention = 0
        
        x = F.interpolate(img, size=self.img_sizes[0], mode="bilinear")  
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed_list[0]   
        keypoint_tokens = self.keypoint_tokens.expand(B, -1, -1)
        quality_tokens = self.quality_token.expand(B, -1, -1)  
        x = torch.cat((quality_tokens, keypoint_tokens, x), dim=1)
        x = self.pos_drop(x)
        embedding_x1 = x    
        for index,blk in enumerate(self.blocks):  
            x, atten = blk(x)
            if index in self.target_index:
                global_attention = self.beta * global_attention + (1-self.beta)*atten
        x = self.norm(x)
        self.global_attention = global_attention
        quality_tokens, keypoint_tokens, feature_temp = torch.split_with_sizes(x, [1, self.num_keypoints, self.patch_embed.num_patches[0]], dim=1)
        results.append(keypoint_tokens)   
        keypoint_tokens_reshaped = keypoint_tokens.permute(0, 2, 1).contiguous().view(B, self.num_keypoints, 24, 32)
        local_detail = self.local_conv(keypoint_tokens_reshaped)  
        global_relation = self.joint_relation(keypoint_tokens_reshaped)  
        enhanced_tokens = (global_relation + local_detail).view(B, self.num_keypoints, -1)
        new_keypoint_tokens = enhanced_tokens  

        # empirical use init embed in next stage
        keypoint_tokens = self.keypoint_tokens.expand(B, -1, -1)  

        quality = self.quality_predictor(quality_tokens)  # 输出为（B，1）
        enable_qp_mask = self.train_epoch is None or not self.training and self.train_epoch > self.qp_start_epoch

        if enable_qp_mask:   
            mask = quality[:, 0] < quality[:, 1] if self.qp_abs else quality[:, 0] < self.qp_threshold
            feature_temp = feature_temp[mask]
            embedding_x1 = embedding_x1[mask]
            keypoint_tokens = keypoint_tokens[mask]
            img = img[mask]
            global_attention = global_attention[mask]
        
        # reuse
        feature_temp = self.reuse_block(feature_temp)  
        B, _, C = feature_temp.shape  
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, *self.patch_embed.patch_shapes[0])  
        feature_temp = F.interpolate(feature_temp, self.patch_embed.patch_shapes[1], mode='nearest')
        feature_temp = feature_temp.view(B, C, self.patch_embed.num_patches[1]).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, self.num_keypoints, self.embed_dim, device=x.device), feature_temp), dim=1)

        x = F.interpolate(img, size=self.img_sizes[1], mode="bilinear")
        x = self.patch_embed(x)
        x = x + self.pos_embed_list[1]
        x = torch.cat((keypoint_tokens, x), dim=1)
        
        embedding_x2 = x + feature_temp
        if self.informative_selection:
            keypoints_attn = global_attention.mean(dim=1)[:, 1:self.num_keypoints, self.num_keypoints+1:].sum(dim=1)
            import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches[0])
            policy_index = torch.argsort(keypoints_attn, dim=1, descending=True)
            unimportan_index = policy_index[:, import_token_num:]
            important_index = policy_index[:, :import_token_num]   
            unimportan_tokens = batch_index_select(embedding_x1, unimportan_index + self.num_keypoints + 1)
            important_index = get_index(important_index, 
                                        patch_shape_src=self.patch_embed.patch_shapes[0],
                                        patch_shape_des=self.patch_embed.patch_shapes[1])
            cls_index = torch.arange(self.num_keypoints, device=x.device).unsqueeze(0).repeat(B, 1)
            important_index = torch.cat((cls_index, important_index + self.num_keypoints), dim=1)
            important_tokens = batch_index_select(embedding_x2, important_index)
            x = torch.cat((important_tokens, unimportan_tokens), dim=1)  
        
        if self.replace_oks:
            quality_tokens = self.quality_token.expand(B, -1, -1)
            x = torch.cat((quality_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)   
        if self.replace_oks:
            quality_tokens = x[:, :1]
            quality_fine = self.quality_predictor(quality_tokens)
            keypoint_tokens = x[:, 1:self.num_keypoints + 1]
        else:
            keypoint_tokens = x[:, :self.num_keypoints]  
        
        if enable_qp_mask:   
            # reassemble tokens
            placeholder = torch.zeros(results[0].shape, device=x.device)
            placeholder[mask] = keypoint_tokens
            placeholder[~mask] = results[0][~mask]
            keypoint_tokens = placeholder
            
            if self.replace_oks:
                quality_placeholder = torch.zeros(quality.shape, device=x.device)
                quality_placeholder[mask] = quality_fine
                quality_placeholder[~mask] = quality[~mask]
                quality_fine = quality_placeholder
       
        if self.replace_oks:  
            quality = [quality[:, 0], quality_fine[:, 0]]
        else:
            quality = [quality[:, 0]]

        results.append(keypoint_tokens)  
        orial_tokens = keypoint_tokens
        keypoint_tokens = self.cross_attention(keypoint_tokens, new_keypoint_tokens)
        connections = torch.tensor([[16, 14], [14, 12], [11, 13], [13, 15], [10, 8], [8, 6], [5, 7], [7, 9]])  # (8, 2)
        edge_index =connections.t().contiguous().clone().detach()  
        edge_index =edge_index.to('cuda')
        skeleton_tokens = self.keypoint_ske(orial_tokens,edge_index)
        ske.append(skeleton_tokens)
        combined_tokens = torch.cat((keypoint_tokens, skeleton_tokens), dim=1)
        new_order = torch.tensor([16, 17, 14, 18, 12, 11, 19, 13, 20, 15, 10, 21, 8, 22, 6, 5, 23, 7, 24, 9, 4, 2, 0, 1, 3])
        combined_tokens = reorder_input(combined_tokens, new_order)
        combined_tokens = self.bilstm(combined_tokens)
        reverse_order = torch.argsort(new_order)
        combined_tokens = recover_output(combined_tokens, reverse_order)
        keypoint_tokens = combined_tokens[:, :17, :]


        results.append(keypoint_tokens)  

        return results, ske,(quality, mask.sum() if enable_qp_mask else B)

