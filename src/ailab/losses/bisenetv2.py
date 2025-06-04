import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMCrossEntropyLoss(nn.Module):
    """在线困难样本挖掘交叉熵损失"""
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000):
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = thresh
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')
    
    def forward(self, score, target):
        # 确保target是3D的 (B, H, W)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # 处理尺寸不匹配的情况
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(score, size=(h, w), mode='bilinear', align_corners=True)
        
        # 计算每个像素的损失
        loss = self.criterion(score, target).view(-1)
        
        # 过滤掉ignore_label的像素
        valid_mask = (target != self.ignore_label).view(-1)
        loss = loss[valid_mask]
        
        if len(loss) == 0:
            return torch.tensor(0.0, device=score.device)
        
        # OHEM: 选择困难样本
        loss_sorted, _ = torch.sort(loss, descending=True)
        
        if len(loss_sorted) < self.min_kept:
            return loss.mean()
        
        if loss_sorted[self.min_kept] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss_sorted[:self.min_kept]
        
        return loss.mean()


class BiSeNetV2Loss(nn.Module):
    """BiSeNetV2的损失函数，支持多种输入格式"""
    
    def __init__(self, ignore_label=255, aux_weight=[1, 0.4, 0.4, 0.4]):
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weight = aux_weight
        
        # 主损失使用OHEM
        self.main_loss = OHEMCrossEntropyLoss(ignore_label)
        # 辅助损失使用普通交叉熵
        self.aux_loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
    
    def forward(self, outputs, target, labels=None, class_mapping=None):
        """
        前向传播
        
        Args:
            outputs: 模型输出
                - 训练模式: tuple (main_output, [aux1, aux2, aux3])
                - 推理模式: tensor (B, C, H, W)
            target: 支持多种格式：
                - tensor (B, H, W): 语义分割标签
                - tensor (B, N, H, W): 批量实例masks
                - tensor (N, H, W): 单个样本的实例masks
                - dict: 包含'masks'和'labels'的字典
            labels: 实例对应的类别标签
            class_mapping: 可选的类别映射函数或字典
        
        Returns:
            loss: 总损失
        """
        # 处理不同的输入格式
        target = self._process_target(outputs, target, labels, class_mapping)
        
        # 计算损失
        if isinstance(outputs, tuple):
            # 训练模式：主输出 + 辅助输出
            main_out, aux_outs = outputs
            
            # 主损失
            main_loss = self.main_loss(main_out, target)
            
            # 辅助损失
            aux_losses = []
            for aux_out in aux_outs:
                aux_losses.append(self.aux_loss(aux_out, target))
            
            # 总损失
            total_loss = self.aux_weight[0] * main_loss
            for i, aux_loss in enumerate(aux_losses):
                total_loss += self.aux_weight[i+1] * aux_loss
            
            if self.training:
                return total_loss
            else:
                return {
                    'total_loss': total_loss,
                    'main_loss': main_loss,
                    'aux_losses': aux_losses
                }
        else:
            # 推理模式：只有主输出
            return self.main_loss(outputs, target)
    
    def _process_target(self, outputs, target, labels=None, class_mapping=None):
        """处理不同格式的目标标签"""
        # 获取输出的形状信息
        if isinstance(outputs, tuple):
            output_shape = outputs[0].shape  # (B, C, H, W)
        else:
            output_shape = outputs.shape
        
        batch_size = output_shape[0]
        height, width = output_shape[2:]
        
        # 处理tensor输入
        if isinstance(target, torch.Tensor):
            if target.dim() == 2:  
                # (H, W) - 单个语义标签
                return target.unsqueeze(0).expand(batch_size, -1, -1)
            
            elif target.dim() == 3:
                if target.shape[0] == batch_size:
                    # (B, H, W) - 已经是语义标签格式
                    return target
                else:
                    # (N, H, W) - 单个样本的实例masks
                    if labels is None:
                        # 如果没有提供labels，使用实例索引作为类别
                        labels = torch.arange(target.shape[0], device=target.device)
                    semantic_label = self.masks_to_semantic_label(target, labels, (height, width))
                    return semantic_label.unsqueeze(0).expand(batch_size, -1, -1)
            
            elif target.dim() == 4:
                # (B, N, H, W) - 批量实例masks
                return self._process_batch_instance_masks(target, labels, (height, width), class_mapping)
            
            else:
                raise ValueError(f"Unsupported target tensor dimension: {target.dim()}")
        
        # 处理字典输入
        elif isinstance(target, dict):
            if 'masks' in target:
                masks = target['masks']
                labels = target.get('labels', None)
                if masks.dim() == 4:  # 批量masks
                    return self._process_batch_instance_masks(masks, labels, (height, width))
                else:  # 单个样本masks
                    semantic_label = self.masks_to_semantic_label(masks, labels, (height, width))
                    return semantic_label.unsqueeze(0).expand(batch_size, -1, -1)
        
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")
    
    def _process_batch_instance_masks(self, masks, labels=None, img_size=None, class_mapping=None):
        """
        处理批量的实例masks
        
        Args:
            masks: tensor of shape (B, N, H, W) - 批量实例masks
            labels: 可选的标签，可以是：
                - None: 使用实例索引作为类别
                - tensor (B, N): 每个样本每个实例的类别
                - list of tensor: 每个样本的类别列表
            img_size: (H, W) 输出尺寸
            class_mapping: 类别映射
        
        Returns:
            semantic_labels: tensor of shape (B, H, W)
        """
        batch_size, max_instances = masks.shape[:2]
        semantic_labels = []
        
        for b in range(batch_size):
            # 获取当前样本的masks
            sample_masks = masks[b]  # (N, H, W)
            
            # 确定有效的实例数（非全零的mask）
            valid_mask_indices = []
            for i in range(max_instances):
                if sample_masks[i].sum() > 0:
                    valid_mask_indices.append(i)
            
            if len(valid_mask_indices) == 0:
                # 没有有效实例，全部为背景
                semantic_label = torch.full(
                    img_size or sample_masks.shape[1:], 
                    self.ignore_label, 
                    dtype=torch.long, 
                    device=masks.device
                )
            else:
                # 获取有效的masks
                valid_masks = sample_masks[valid_mask_indices]
                
                # 获取对应的标签
                if labels is None:
                    # 使用实例索引作为类别（从0开始）
                    sample_labels = torch.tensor(valid_mask_indices, device=masks.device)
                elif isinstance(labels, torch.Tensor):
                    if labels.dim() == 2:  # (B, N)
                        sample_labels = labels[b][valid_mask_indices]
                    else:  # (N,) - 所有样本共享
                        sample_labels = labels[valid_mask_indices]
                elif isinstance(labels, list):
                    sample_labels = labels[b]
                else:
                    sample_labels = torch.tensor(valid_mask_indices, device=masks.device)
                
                # 应用类别映射
                if class_mapping is not None:
                    if callable(class_mapping):
                        sample_labels = torch.tensor(
                            [class_mapping(l.item()) for l in sample_labels],
                            device=masks.device
                        )
                    elif isinstance(class_mapping, dict):
                        sample_labels = torch.tensor(
                            [class_mapping.get(l.item(), l.item()) for l in sample_labels],
                            device=masks.device
                        )
                
                # 转换为语义标签
                semantic_label = self.masks_to_semantic_label(valid_masks, sample_labels, img_size)
            
            semantic_labels.append(semantic_label)
        
        return torch.stack(semantic_labels)
    
    def masks_to_semantic_label(self, masks, labels=None, img_size=None):
        """
        将实例分割masks转换为语义分割标签
        
        Args:
            masks: tensor of shape (N, H, W)
            labels: tensor of shape (N,) or None
            img_size: (H, W) 输出尺寸
        
        Returns:
            semantic_label: tensor of shape (H, W)
        """
        if len(masks) == 0:
            if img_size is None:
                raise ValueError("img_size must be provided when masks is empty")
            return torch.full(img_size, self.ignore_label, dtype=torch.long, device=masks.device)
        
        h, w = masks.shape[1:]
        if img_size is None:
            img_size = (h, w)
        
        # 如果没有提供labels，使用索引
        if labels is None:
            labels = torch.arange(len(masks), device=masks.device)
        
        # 初始化语义标签
        device = masks.device
        semantic_label = torch.full((h, w), self.ignore_label, dtype=torch.long, device=device)
        
        # 处理每个实例
        for mask, label in zip(masks, labels):
            semantic_label[mask > 0.5] = label.item() if label.dim() == 0 else int(label)
        
        # 调整大小
        if (h, w) != img_size:
            semantic_label = F.interpolate(
                semantic_label.unsqueeze(0).unsqueeze(0).float(),
                size=img_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        
        return semantic_label
