import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MSELoss, KLDivLoss

from mmpose.models.builder import HEADS, build_loss
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead

from einops import rearrange, repeat
from .eval_utils import pose_pck_accuracy, instance_pck_accuracy, instance_oks
from mmpose.core.post_processing import flip_back


@HEADS.register_module()
class SHaRPoseQPHeatmapHeadLstm(TopdownHeatmapBaseHead):
    def __init__(self,
                 in_channels=768,
                 heatmap_size=(64, 48),
                 loss_keypoint=None,
                 loss_distill=None,
                 multi_layer=None,
                 replace_oks=False,
                 qp_type='oks',
                 qp_start_epoch=90,
                 qp_abs=False,
                 qp_threshold=0.99,
                 train_cfg=None,
                 test_cfg=None,
                 offset_loss_weight=0.05,
                 skele_loss_weight=0.01,      
                 use_offset_refine=True,       
                 ):
        super().__init__()
        
        self.loss = build_loss(loss_keypoint)
        
        if loss_distill == 'kl':
            loss_distill = KLDivLoss()
        elif loss_distill == 'mse':
            loss_distill = MSELoss()
        else:
            loss_distill = None
        self.loss_distill = loss_distill

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        
        self.heatmap_size = heatmap_size
        self.heatmap_dim = heatmap_size[0] * heatmap_size[1]
        
        if (multi_layer is None and in_channels >= self.heatmap_dim // 4) or multi_layer == False:
            self.heatmap_mlp = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, self.heatmap_dim)
            )
        else:
            self.heatmap_mlp = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, self.heatmap_dim // 2),
                nn.GELU(),
                nn.LayerNorm(self.heatmap_dim // 2),
                nn.Linear(self.heatmap_dim // 2, self.heatmap_dim)
            )
            
        self.qp_start_epoch = qp_start_epoch
        self.qp_type = qp_type
        self.qp_abs = qp_abs
        self.qp_threshold = qp_threshold
        self.train_epoch = None
        self.replace_oks = replace_oks
        self.default_eval_stage = -1
        self.loss2 = SkeletonLoss(loss_type='BCE', loss_weight=1.0)
        self.offset_fc = nn.Sequential(
            nn.Linear(in_channels, 2),
            nn.Sigmoid()          # 输出范围 [0,1]
        )
        self.offset_loss_weight = offset_loss_weight
        self.skele_loss_weight = skele_loss_weight
        self.use_offset_refine = use_offset_refine

    def forward(self, output):
        """
        x: keypoint tokens [B N C][64.17,768]
        """
        x, y, (self.quality, self.n_refine) = output
        if isinstance(x, list):
            self.embedding_outputs = []
            result = []
            for tk in x:
                self.embedding_outputs.append(tk)
                tk = self.heatmap_mlp(tk)
                tk = rearrange(tk, "b n (h w) -> b n h w",
                            h=self.heatmap_size[0],
                            w=self.heatmap_size[1])
                result.append(tk)
        else:
            x = self.heatmap_mlp(x)
            result = rearrange(x, "b n (h w) -> b n h w",
                            h=self.heatmap_size[0],
                            w=self.heatmap_size[1])
        if self.replace_oks:
            self.maxval = self.quality[-1][:, None, None].repeat(1, result[-1].shape[1], 1) \
                            .detach().cpu()
        if isinstance(y, list):
            self.embedding_outputs_y = []
            ske = []
            for tk in y:
                self.embedding_outputs_y.append(tk)
                tk = self.heatmap_mlp(tk)
                tk = rearrange(tk, "b n (h w) -> b n h w",
                            h=self.heatmap_size[0],
                            w=self.heatmap_size[1])
                ske.append(tk)
        else:
            y = self.heatmap_mlp(y)
            ske = rearrange(y, "b n (h w) -> b n h w",
                            h=self.heatmap_size[0],
                            w=self.heatmap_size[1])
        return result, ske

    def get_loss(self, output, target, target_weight, img_metas):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W
            img_metas：图像元数据的列表，缩放，中心位置等（用于计算OKS）

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        result, ske = output

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        if isinstance(self.quality, torch.Tensor):
            self.quality = [self.quality]
        if isinstance(result, list):
            for i, hm in enumerate(result):
                losses[f'heatmap_loss_{i}'] = self.loss(hm, target, target_weight)
                if i in [0, 1][:int(self.replace_oks) + 1]:
                    factor = 0.03 if self.train_epoch is not None and self.train_epoch > self.qp_start_epoch else 0.0
                    if self.qp_type == 'pck':
                        pck, mask = instance_pck_accuracy(hm.detach().cpu().numpy(),
                                                        target.detach().cpu().numpy(),
                                                        target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                        pck = torch.from_numpy(pck).cuda().float()
                        mask = torch.from_numpy(mask).cuda()
                        if self.qp_abs:
                            qp_loss = factor * F.cross_entropy(self.quality[i][mask],
                                                               (pck > self.qp_threshold).to(dtype=int)[mask])
                        else:
                            qp_loss = factor * F.mse_loss(self.quality[i][mask], pck)
                        losses[f'pck_{i}'] = pck.mean()
                    elif self.qp_type == 'oks':
                        oks, mask = instance_oks(hm.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy(),
                                                 np.stack([meta['scale'] for meta in img_metas]),
                                                 np.stack([meta['center'] for meta in img_metas]),
                                                 target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                        oks = torch.from_numpy(oks).cuda().float()
                        mask = torch.from_numpy(mask).cuda()
                        if self.qp_abs:
                            qp_loss = factor * F.cross_entropy(self.quality[i][mask],
                                                               (oks > self.qp_threshold).to(dtype=int)[mask])
                        else:
                            qp_loss = factor * F.mse_loss(self.quality[i][mask], oks)
                        losses[f'oks_{i}'] = oks.mean()
                    if i == 0:
                        losses[f'refine_rate'] = self.n_refine / hm.shape[0]
                    losses[f'qp_loss_{i}'] = qp_loss
            if self.loss_distill is not None:
                for i, embed in enumerate(self.embedding_outputs[:-1]):
                    losses[f'distill_loss_{i}'] = self.loss_distill(embed, self.embedding_outputs[-1].detach())
        else:
            losses['heatmap_loss'] = self.loss(result, target, target_weight)
            
        if self.use_offset_refine:
            # 获取最终阶段的 keypoint tokens
            final_tokens = self.embedding_outputs[-1]          # (B, K, C)
            pred_coords = self.offset_fc(final_tokens)         # (B, K, 2)

            # 从 img_metas 提取 GT 归一化坐标（浮点精度）
            heatmap_size = (self.heatmap_size[0], self.heatmap_size[1])  # (W, H)
            gt_coords = self._get_gt_coords_from_metas(img_metas, heatmap_size)
            gt_coords = gt_coords.to(target.device)            # 确保设备一致

            # 权重处理
            if target_weight.dim() == 2:
                weight = target_weight.unsqueeze(-1)           # (B, K, 1)
            elif target_weight.dim() == 3:
                weight = target_weight
            else:
                raise ValueError(f"Unexpected target_weight dim: {target_weight.dim()}")

            weight_expand = weight.expand(-1, -1, 2)           # (B, K, 2)

            # Smooth L1 Loss
            loss_offset = F.smooth_l1_loss(pred_coords, gt_coords, reduction='none')
            loss_offset = (loss_offset * weight_expand).sum() / (weight_expand.sum() + 1e-8)

            losses['offset_loss'] = self.offset_loss_weight * loss_offset
        # 骨架损失（原有）
        skeleton_pairs = [(16, 14), (14, 12), (11, 13), (13, 15), (10, 8), (8, 6), (5, 7), (7, 9)]
        skeleton_target = self._generate_skeleton_heatmap(target, skeleton_pairs)
        skeleton_target = skeleton_target.to('cuda')
        if isinstance(ske, list):
            total_skele_loss = 0
            for i, hm in enumerate(ske):
                # total_skele_loss += self.loss2(hm, skeleton_target) * 0.01
                total_skele_loss += self.loss2(hm, skeleton_target) * self.skele_loss_weight
            losses['skele_loss'] = total_skele_loss
        else:
            # losses['skele_loss'] = self.loss2(ske, skeleton_target) * 0.01
            losses['skele_loss'] = self.loss2(ske, skeleton_target) * self.skele_loss_weight

        return losses


    def _get_gt_coords_from_metas(self, img_metas, heatmap_size):
        W_hm, H_hm = heatmap_size
        gt_coords_list = []

        for meta in img_metas:
            joints = meta['joints_3d']          # (K, 3) numpy
            center = meta['center']             # (2,) numpy
            scale = meta['scale']               # scalar or (2,) numpy/list

            # 全部转为 torch tensor
            if isinstance(joints, np.ndarray):
                joints = torch.from_numpy(joints).float()
            else:
                joints = torch.tensor(joints, dtype=torch.float)
            if isinstance(center, np.ndarray):
                center = torch.from_numpy(center).float()
            else:
                center = torch.tensor(center, dtype=torch.float)
            if isinstance(scale, (int, float)):
                scale = torch.tensor([scale, scale], dtype=torch.float)
            elif isinstance(scale, np.ndarray):
                scale = torch.from_numpy(scale).float()
            else:
                scale = torch.tensor(scale, dtype=torch.float)  # list 或 tuple

            pts = joints[:, :2]   # (K, 2)

            # 逆 UDP 仿射变换
            x_hm = (pts[:, 0] - center[0]) / scale[0] * (W_hm - 1) / 2.0 + (W_hm - 1) / 2.0
            y_hm = (pts[:, 1] - center[1]) / scale[1] * (H_hm - 1) / 2.0 + (H_hm - 1) / 2.0

            x_norm = x_hm / (W_hm - 1)
            y_norm = y_hm / (H_hm - 1)

            x_norm = torch.clamp(x_norm, 0.0, 1.0)
            y_norm = torch.clamp(y_norm, 0.0, 1.0)

            coords = torch.stack([x_norm, y_norm], dim=1)   # (K, 2)
            gt_coords_list.append(coords)

        return torch.stack(gt_coords_list, dim=0)    

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        accuracy = dict()
        result, ske = output
        
        if self.target_type == 'GaussianHeatmap':
            if isinstance(result, list):
                for i, hm in enumerate(result):
                    acc, avg_acc, _ = pose_pck_accuracy(
                        hm.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                        target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                    accuracy[f'acc_pose_{i}'] = float(avg_acc)
            else:
                _, avg_acc, _ = pose_pck_accuracy(
                        result.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                        target_weight.detach().cpu().numpy().squeeze(-1) > 0)
                accuracy['acc_pose'] = float(avg_acc)
        return accuracy
    
    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, list):
            output = output[self.default_eval_stage]

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap
    
    def init_weights(self):
        """Initialize model weights."""
        pass
        
    def decode(self, img_metas, output, **kwargs):
        result = super().decode(img_metas, output, **kwargs)
        if self.replace_oks:
            result['preds'][:, :, 2:3] = self.maxval
        return result

    def _generate_skeleton_heatmap(self, keypoints_heatmap, skeleton_pairs):
        """
        生成骨架热力图
        :param keypoints_heatmap: 关键点的热力图，形状为 (N, K, H, W)
        :param skeleton_pairs: 关键点对列表，表示骨架连接，每个元素是一个 (start, end) 元组
        :return: 骨架热力图，形状为 (N, L, H, W)，其中 L 是骨架连接数
        """
        batch_size, num_joints, height, width = keypoints_heatmap.shape
        skeleton_heatmap = torch.zeros(batch_size, len(skeleton_pairs), height, width)

        for i, (start, end) in enumerate(skeleton_pairs):
            start_heatmap = keypoints_heatmap[:, start, :, :]
            end_heatmap = keypoints_heatmap[:, end, :, :]
            skeleton_heatmap[:, i, :, :] = (start_heatmap + end_heatmap) / 2

        return skeleton_heatmap


class SkeletonLoss(nn.Module):
    def __init__(self, loss_type='BCE', loss_weight=1.0):
        super(SkeletonLoss, self).__init__()
        assert loss_type in ['BCE', 'MSE'], "Loss type must be 'BCE' or 'MSE'"
        self.loss_type = loss_type
        self.loss_weight = loss_weight

        if self.loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self.loss_type == 'MSE':
            self.criterion = nn.MSELoss(reduction='none')

    def forward(self, predicted_skeleton, target_heatmaps):
        batch_size, num_skeletons, H, W = predicted_skeleton.size()
        skeleton_loss = 0.0

        for i in range(num_skeletons):
            target_skeleton = target_heatmaps[:, i, :, :]
            pred_skeleton = predicted_skeleton[:, i, :, :]

            if self.loss_type == 'BCE':
                loss = self.criterion(pred_skeleton, target_skeleton)
            elif self.loss_type == 'MSE':
                loss = self.criterion(pred_skeleton, target_skeleton)

            skeleton_loss += loss.mean()

        skeleton_loss = skeleton_loss / num_skeletons * self.loss_weight
        return skeleton_loss