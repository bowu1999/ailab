import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


class ProtoClassifier:
    """
    基于原型的分类器，支持动态更新原型集。
    
    参数:
    num_classes (int): 分类任务中的类别数量。
    feat_dim (int): 输入特征的维度。
    k_init (int, optional): 每个类别的初始原型数量，默认值为5。
    k_max (int, optional): 每个类别最大允许的原型数量，默认值为10。
    """    
    def __init__(self, num_classes, feat_dim, k_init=5, k_max=10):
        self.num_classes = num_classes  # 类别总数
        self.k_init = k_init  # 初始时每个类别的原型数
        self.k_max = k_max  # 每个类别最多能有的原型数
        # 初始化原型列表，每个元素是一个形状为 (k_i, feat_dim) 的数组，代表第i类的所有原型
        self.prototypes = [None] * num_classes
    
    def fit(self, X, y):
        """
        使用训练数据拟合分类器，并初始化每个类别的原型集。
        
        参数:
        X (np.ndarray): 训练样本特征矩阵，形状为 (n_samples, feat_dim)。
        y (np.ndarray): 对应的标签向量，长度为 n_samples。
        """
        for c in range(self.num_classes):
            # 获取属于当前类别的所有样本
            Xc = X[y == c]
            if len(Xc) > 0:  # 只有当存在样本时才进行聚类
                k = min(self.k_init, len(Xc))
                km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
                self.prototypes[c] = km.fit(Xc).cluster_centers_
            else:
                pass
    
    def predict(self, X):
        """
        预测输入样本所属的类别。
        
        参数:
        X (np.ndarray): 待预测样本特征矩阵，形状为 (n_samples, feat_dim)。
        
        返回:
        np.ndarray: 预测结果，长度为 n_samples 的一维数组，每个元素是预测的类别索引。
        """
        dists = []  # 存储每类到样本的距离
        for c, pcs in enumerate(self.prototypes):
            # 计算当前类别所有原型与待预测样本之间的欧氏距离
            d = euclidean_distances(X, pcs)
            # 对于每个样本，选择到最近原型的距离作为该样本到该类别的距离
            dists.append(d.min(axis=1))
        # 找出每个样本距离最近的类别索引
        return np.argmin(np.stack(dists, axis=1), axis=1)
    
    def partial_update(self, X_new, y_new):
        """
        根据新到达的数据部分更新原型集。
        
        参数:
        X_new (np.ndarray): 新到达的样本特征矩阵。
        y_new (np.ndarray): 对应的新到达样本的标签。
        """
        # 对于新到达数据中的每个类别
        for c in np.unique(y_new):
            # 获取属于当前类别的所有新样本
            Xc = X_new[y_new == c]
            pcs = self.prototypes[c]  # 当前类别的原型集
            # 将新样本与现有原型合并
            all_pts = np.vstack([pcs, Xc])
            # 确定更新后该类别的原型数k，不超过k_max且不超过合并后的总样本数
            k = min(self.k_max, all_pts.shape[0])
            # 使用MiniBatchKMeans算法重新聚类以更新原型集
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
            self.prototypes[c] = km.fit(all_pts).cluster_centers_


def extract_image_features_and_labels(model, dataset, output_dir, batch_size=32, num_workers=8):
    """
    提取给定数据集中的所有特征和标签，并将它们保存为npy文件。
    支持数据集返回tuple(image, label)或dict(image=image, label=label)。
    
    参数:
        model: 预训练的PyTorch模型，用于特征提取
        dataset: 自定义的数据集类实例，包含图像路径和分类标签
        output_dir: 输出目录，用于保存提取的特征和标签
        batch_size: 数据加载器的批量大小
        num_workers: 数据加载器的工作线程数
    """
    # 设置模型为评估模式
    model.eval()
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    # 初始化列表来存储特征和标签
    features_list = []
    labels_list = []
    # 禁用梯度计算
    with torch.no_grad():
        for batch in dataloader:
            # 检查batch是否为字典或元组
            if isinstance(batch, dict):
                images = batch['image'].to(next(model.parameters()).device)  # 将图像移动到与模型相同的设备上
                label = batch['label'].cpu().numpy()  # 获取分类标签并转换为numpy数组
            elif isinstance(batch, tuple) and len(batch) == 2:
                images, label = batch
                images = images.to(next(model.parameters()).device)  # 将图像移动到与模型相同的设备上
                label = label.cpu().numpy()  # 获取分类标签并转换为numpy数组
            else:
                raise ValueError(
                    "Batch format not supported. Batch must be a dictionary or a tuple of (images, labels)."
                )
            # 前向传播获取特征
            feats = model(images).cpu().numpy()
            # 添加到列表中
            features_list.append(feats)
            labels_list.append(label)
    # 将所有批次的特征和标签合并
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # 保存特征和标签
    np.save(Path(output_dir) / 'features.npy', all_features)
    np.save(Path(output_dir) / 'labels.npy', all_labels)