import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render,
)

# isort: split
from fusion_module import CrossModalFusionModule
from fusion_module1 import CrossModalFusionModule1
from image_backbone import FeaturePyramid, ImageBackbone
from point_backbone import PointBackbone
from utils import get_2d3d_node_correspondences, patchify
from vision3d.ops import mutual_topk_select, pairwise_distance
from torch.autograd import Function

class SuperPointMatchingMutualTopk_myself(SuperPointMatchingMutualTopk):
    def __init__(self, num_correspondences, k, threshold=None, mutual=True, eps=1e-8):
        super().__init__(num_correspondences, k, threshold=None, mutual=True, eps=1e-8)
        self.num_correspondences = num_correspondences
        self.k = k
        self.threshold = threshold
        self.mutual = mutual
        self.eps = eps
    def forward(self, src_feats, tgt_feats1 ,tgt_feats2,tgt_feats3, src_masks=None, tgt_masks=None):
     
        if src_masks is not None:
            src_feats = src_feats[src_masks]

        if tgt_masks is not None:
            tgt_feats1 = tgt_feats1[tgt_masks]
            tgt_feats2 = tgt_feats2[tgt_masks]
            tgt_feats3 = tgt_feats3[tgt_masks]
              
        score_mat1 = torch.sqrt(pairwise_distance(src_feats, tgt_feats1, normalized=True) + self.eps)
        score_mat2 = torch.sqrt(pairwise_distance(src_feats, tgt_feats2, normalized=True) + self.eps)
        score_mat3 = torch.sqrt(pairwise_distance(src_feats, tgt_feats3, normalized=True) + self.eps)
        score_mat4  = torch.cat([score_mat1.unsqueeze(0), score_mat2.unsqueeze(0), score_mat3.unsqueeze(0)], dim=0)
        score_mat, score_mat_indices = torch.max(score_mat4,dim=0)

        src_corr_indices, tgt_corr_indices, corr_scores = mutual_topk_select(
            score_mat, self.k, largest=False, threshold=None, mutual=self.mutual
        )

        # threshold
        if self.threshold is not None:
            num_correspondences = min(self.num_correspondences, corr_scores.numel())
            masks = torch.le(corr_scores, self.threshold)
            # print(masks.sum().item())
            if masks.sum().item() < num_correspondences:
                # not enough good correspondences, fallback to topk selection
                corr_scores, topk_indices = corr_scores.topk(k=num_correspondences, largest=False)
                src_corr_indices = src_corr_indices[topk_indices]
                tgt_corr_indices = tgt_corr_indices[topk_indices]
            else:
                src_corr_indices = src_corr_indices[masks]
                tgt_corr_indices = tgt_corr_indices[masks]
                corr_scores = corr_scores[masks]

        # recover original indices
        if src_masks is not None:
            src_valid_indices = torch.nonzero(src_masks, as_tuple=True)[0]
            src_corr_indices = src_valid_indices[src_corr_indices]

        if tgt_masks is not None:
            tgt_valid_indices = torch.nonzero(tgt_masks, as_tuple=True)[0]
            tgt_corr_indices = tgt_valid_indices[tgt_corr_indices]

        return src_corr_indices, tgt_corr_indices, corr_scores

class Middlelayer(nn.Module):
    def __init__(self, GRL = None):
        super(Middlelayer, self).__init__()
        if GRL:
            self.grl = GRL()

    def forward(self, x):
        if getattr(self, 'grl', None) is not None:
            x = self.grl(x)
        return x
    
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)   # 第二个全连接层
        self.fc3 = nn.Linear(64, 2)     # 输出层
       
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    



class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff=1.0) -> torch.Tensor:
        ctx.coeff = coeff
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None

class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, input: torch.Tensor, coeff: float = 0.01):
        return GradientReverseFunction.apply(input, coeff)

class B23D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        # fixed for now
        self.img_h_c = 24
        self.img_w_c = 32
        self.img_num_levels_c = 3
        self.overlap_threshold = 0.3  #重叠阈值，一般0.3就认为重叠
        self.pcd_min_node_size = 5   #点云最小节点大小

        self.img_backbone = ImageBackbone(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim,
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation,
        )

        self.pcd_backbone = PointBackbone(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim,
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma,
        )

        self.transformer = CrossModalFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding,
        )
        
        self.crosstransformer = CrossModalFusionModule1(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding,
        )

        self.i0_conv = nn.Conv2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=(24, 32),
                            bias=True
                            )
        nn.init.zeros_(self.i0_conv.bias)#conv2d

        self.i1_conv = nn.Conv2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=(12, 16),
                            bias=True
                            )
        nn.init.zeros_(self.i1_conv.bias)#conv2d 

        self.middlelayer = Middlelayer(GRL = GRL)
        self.classifier = SimpleClassifier()
        
        self.img_pyramid = FeaturePyramid(cfg.model.transformer.output_dim)

        self.coarse_target = SuperPointProposalGenerator(
            cfg.model.coarse_matching.num_targets,
            cfg.model.coarse_matching.overlap_threshold,
        )

        self.coarse_matching = SuperPointMatchingMutualTopk_myself(
            cfg.model.coarse_matching.num_correspondences,
            k=cfg.model.coarse_matching.topk,
            threshold=cfg.model.coarse_matching.similarity_threshold,
        )

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."

        torch.cuda.synchronize()
        start_time = time.time()

        output_dict = {}

        # 1. Unpack data

        # 1.1 Unpack 2D data
        image = data_dict["image"].unsqueeze(1).detach()  # (B, 1, H, W), gray scaling_factor
        depth = data_dict["depth"].detach()  # (B, H, W)
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = data_dict["transform"].detach()

        img_h = image.shape[2]
        img_w = image.shape[3]
        img_h_f = img_h
        img_w_f = img_w

        img_points, img_masks = back_project(depth, intrinsics, depth_limit=6.0, transposed=True, return_mask=True)
        img_points = img_points.squeeze(0)  # (B, H, W, 3) -> (H, W, 3)
        img_masks = img_masks.squeeze(0)  # (B, H, W) -> (H, W)
        img_pixels = create_meshgrid(img_h, img_w).float()  # (H, W, 2)

        img_points_f = img_points  # (H, H, 3)
        img_masks_f = img_masks  # (H, H)
        img_pixels_f = img_pixels  # (H, W, 2)

        img_points = img_points.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels = img_pixels.view(-1, 2)  # (H, W, 2) -> (HxW, 2)
        img_masks = img_masks.view(-1)  # (H, W) -> (HxW)
        img_points_f = img_points_f.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels_f = img_pixels_f.view(-1, 2)  # (H/2xW/2, 2)
        img_masks_f = img_masks_f.view(-1)  # (H, W) -> (HxW)

        output_dict["img_points"] = img_points
        output_dict["img_pixels"] = img_pixels
        output_dict["img_masks"] = img_masks
        output_dict["img_points_f"] = img_points_f
        output_dict["img_pixels_f"] = img_pixels_f
        output_dict["img_masks_f"] = img_masks_f

        # 1.2 Unpack 3D data
        pcd_feats = data_dict["feats"].detach()
        pcd_points = data_dict["points"][0].detach()
        pcd_points_f = data_dict["points"][0].detach()
        pcd_points_c = data_dict["points"][-1].detach()
        pcd_pixels_f = render(pcd_points_f, intrinsics, extrinsics=transform, rounding=False) #点云坐标投影到图像平面

        output_dict["pcd_points"] = pcd_points
        output_dict["pcd_points_c"] = pcd_points_c
        output_dict["pcd_points_f"] = pcd_points_f
        output_dict["pcd_pixels_f"] = pcd_pixels_f

        # 2. Backbone

        # 2.1 Image backbone
        img_feats_list = self.img_backbone(image)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80) 最后一层特征
        img_feats_f = img_feats_list[0]  # (B, C2, H, W), aka, (1, 128, 480, 640) 第一层特征

        # 2.2 Point backbone
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 1024) 最后一层特征
        pcd_feats_f = pcd_feats_list[0]  # (Nf, 128)  第一层特征
 
        # 3. Transformer

        # 3.1 Prepare image features
        img_shape_c = (self.img_h_c, self.img_w_c)
        img_feats_c = F.interpolate(img_feats_x, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        #计算了新的图像形状 img_shape_c，并将图像特征 img_feats_x 调整为该形状，通过双线性插值将其大小调整为 img_shape_c，这通常用于将不同大小的特征层对齐到相同的大小
        img_feats_c = img_feats_c.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)  # (768, 512)
        #将调整后的图像特征 img_feats_c 重新组织为一个二维矩阵，其中每一行表示一个像素位置，每一列表示一个特征维度
        img_pixels_c = create_meshgrid(self.img_h_c, self.img_w_c, normalized=True, flatten=True)  # (768, 2)
        #使用 create_meshgrid 函数生成了一个网格 img_pixels_c，该网格包含了图像特征矩阵中每个像素的坐标信息，并且已经被展平为一个二维矩阵。
        # use normalized pixel coordinates for transformer

        # 3.2 Cross-modal fusion transformer
        img_feats_c, pcd_feats_c = self.transformer(
            img_feats_c.unsqueeze(0),
            #img_feats_c.unsqueeze(0)：经过处理的图像特征，通过 unsqueeze(0) 方法在批量维度上添加了一个维度，将其形状变为 (1, N, C)，其中 N 表示特征数量，C 表示特征维度。（1，768，512）
            img_pixels_c.unsqueeze(0),
            #img_pixels_c.unsqueeze(0)：经过处理的图像像素坐标，也通过 unsqueeze(0) 方法在批量维度上添加了一个维度，将其形状变为 (1, N, 2)，其中 N 表示像素数量，2 表示坐标维度（x 和 y）。（1，768，2）
            pcd_feats_c.unsqueeze(0),
            #pcd_feats_c.unsqueeze(0)：经过处理的点云特征，同样通过 unsqueeze(0) 方法在批量维度上添加了一个维度，将其形状变为 (1, M, D)，其中 M 表示点云数量，D 表示特征维度。（1，Nc，1024）
            pcd_points_c.unsqueeze(0),
            #pcd_points_c.unsqueeze(0)：经过处理的点云坐标，同样通过 unsqueeze(0) 方法在批量维度上添加了一个维度，将其形状变为 (1, M, 3)，其中 M 表示点云数量，3 表示坐标维度（x、y 和 z）（1，Nf，128）。
        )

        # 3.3 Post-transformer image feature pyramid
        img_feats_c = img_feats_c.transpose(1, 2).contiguous().view(1, -1, self.img_h_c, self.img_w_c)
        #img_feats_c.transpose(1, 2)==(1,512,768)  contiguous保证存储连续性  
        #如果原始张量的形状是 (B, C, H, W)，其中 B 是批量大小，C 是通道数，H 是高度，W 是宽度。经过 view(1, -1, self.img_h_c, self.img_w_c) 操作后，
        #张量的形状变成了 (1, N, self.img_h_c, self.img_w_c)，其中 N 是根据其他维度的大小自动计算得出的，以使得张量的元素总数保持不变。
        all_img_feats_c = self.img_pyramid(img_feats_c)
        all_img_feats_c = [x.squeeze(0).view(x.shape[1], -1).transpose(0, 1).contiguous() for x in all_img_feats_c]
        #经过特征提取后恢复
        img_feats_c = torch.cat(all_img_feats_c, dim=0)

        # 3.4 Post-processing for point features
        pcd_feats_c = pcd_feats_c.squeeze(0)#（NC，1024）

        all_feats = torch.cat((pcd_feats_c, img_feats_c), dim=0)
        pcd_labels = torch.zeros(pcd_feats_c.shape[0], dtype=torch.long)
        img_labels = torch.ones(img_feats_c.shape[0], dtype=torch.long)
        all_labels = torch.cat((pcd_labels, img_labels), dim=0).cuda()

        all_featsm = self.middlelayer(all_feats)

        class_preds = self.classifier(all_featsm)
        output_dict["class_preds"] = class_preds
        output_dict["all_labels"] = all_labels

        img_feats_c = F.normalize(img_feats_c, p=2, dim=1)
        # pcd_feats_c = F.normalize(pcd_feats_c, p=2, dim=1)

        output_dict["img_feats_c"] = img_feats_c
        output_dict["pcd_feats_c"] = pcd_feats_c


        # pcd_feats_c = pcd_feats_c.squeeze(0)#（NC，1024）

        # # img_feats_c = F.normalize(img_feats_c, p=2, dim=1)
        # # pcd_feats_c = F.normalize(pcd_feats_c, p=2, dim=1)

        # output_dict["img_feats_c"] = img_feats_c
        # output_dict["pcd_feats_c"] = pcd_feats_c


        # 4. Coarse-level matching
         

        # 4.1 Generate 3d patches
        _, pcd_node_sizes, pcd_node_masks, pcd_node_knn_indices, pcd_node_knn_masks = point_to_node_partition(
            pcd_points_f, pcd_points_c, self.pcd_num_points_in_patch, gather_points=True, return_count=True)
        #使用 point_to_node_partition 函数对点云进行节点分割，将点云分成不同的节点。
        pcd_node_masks = torch.logical_and(pcd_node_masks, torch.gt(pcd_node_sizes, self.pcd_min_node_size))#用于过滤掉太小的点云节点。
        pcd_padded_points_f = torch.cat([pcd_points_f, torch.ones_like(pcd_points_f[:1]) * 1e10], dim=0)
        #具体来说，它首先创建了一个与 pcd_points_f 的第一个维度相同大小的张量，其中的元素都是 1，并乘以一个很大的数值（1e10），以确保填充的值远远大于原始的点云坐标值。
        #然后，它将这个填充张量与原始的点云坐标张量在第一个维度上进行拼接，得到一个填充后的点云坐标张量 pcd_padded_points_f。
        #填充的作用可能是为了在进行 K 最近邻搜索等操作时，对节点之外的点进行处理。例如，在进行 K 最近邻搜索时，可以将填充后的值视为无效值，从而保证在搜索过程中不考虑填充的点
        pcd_node_knn_points = index_select(pcd_padded_points_f, pcd_node_knn_indices, dim=0)
        #为了在进行 K 最近邻搜索时，根据每个节点的 KNN 索引从点云中选择对应的最近邻点
        #可以根据 KNN 索引从填充后的点云坐标张量中选择对应的节点，并将这些节点的坐标组成一个新的张量 pcd_node_knn_points 
        pcd_padded_pixels_f = torch.cat([pcd_pixels_f, torch.ones_like(pcd_pixels_f[:1]) * 1e10], dim=0)
        #点云投影的像素层面进行填充
        pcd_node_knn_pixels = index_select(pcd_padded_pixels_f, pcd_node_knn_indices, dim=0)
        #选出knn的节点

        # 4.2 Generate 2d patches
        all_img_node_knn_points = []
        all_img_node_knn_pixels = []
        all_img_node_knn_indices = []
        all_img_node_knn_masks = []
        all_img_node_masks = []
        all_img_node_levels = []
        all_img_num_nodes = []
        all_img_total_nodes = []
        total_img_num_nodes = 0

        all_gt_img_node_corr_levels = []
        all_gt_img_node_corr_indices = []
        all_gt_pcd_node_corr_indices = []
        all_gt_img_node_corr_overlaps = []
        all_gt_pcd_node_corr_overlaps = []

        img_h_c = self.img_h_c
        img_w_c = self.img_w_c
        for i in range(self.img_num_levels_c):
            (
                img_node_knn_points,  # (N, Ki, 3)
                img_node_knn_pixels,  # (N, Ki, 2)
                img_node_knn_indices,  # (N, Ki)存储了每个图像节点的KNN索引，即每个图像节点对应的邻居点在原始图像中的索引。这个索引是在原始图像中的二维坐标，用于指示每个图像节点的邻居点的位置
                img_node_knn_masks,  # (N, Ki)存储了每个图像节点的KNN掩码，用于指示每个图像节点的邻居点是否有效或者存在。在处理图像数据时，由于图像边界等因素，
                #某些图像节点可能没有足够的邻居点，或者邻居点的数量可能不同。因此，通过KNN掩码可以标记哪些邻居点是有效的，哪些是无效的。
                img_node_masks,  # (N)
            ) = patchify(img_points_f, img_pixels_f, img_masks_f, img_h_f, img_w_f, img_h_c, img_w_c, stride=2)

            img_num_nodes = img_h_c * img_w_c
            #当前尺度下图像节点的数量 img_num_nodes，即将图像分割成的块或区域的总数。具体来说，它是通过将图像的高度 img_h_c 乘以宽度 img_w_c 得到的
            img_node_levels = torch.full(size=(img_num_nodes,), fill_value=i, dtype=torch.long).cuda()
            #即有 img_num_nodes 个元素。然后使用 torch.full 函数将所有元素填充为当前尺度的索引 i，表示这些节点都属于当前尺度


            all_img_node_knn_points.append(img_node_knn_points)
            all_img_node_knn_pixels.append(img_node_knn_pixels)
            all_img_node_knn_indices.append(img_node_knn_indices)
            all_img_node_knn_masks.append(img_node_knn_masks)
            all_img_node_masks.append(img_node_masks)
            all_img_node_levels.append(img_node_levels)
            all_img_num_nodes.append(img_num_nodes)
            all_img_total_nodes.append(total_img_num_nodes)

            # 4.3 Generate coarse-level ground truth
            (
                gt_img_node_corr_indices,
                gt_pcd_node_corr_indices,
                gt_img_node_corr_overlaps,
                gt_pcd_node_corr_overlaps,
            ) = get_2d3d_node_correspondences(
                img_node_masks,
                img_node_knn_points,
                img_node_knn_pixels,
                img_node_knn_masks,
                pcd_node_masks,
                pcd_node_knn_points,
                pcd_node_knn_pixels,
                pcd_node_knn_masks,
                transform,
                self.matching_radius_2d,
                self.matching_radius_3d,
            )
            # gt_img_node_corr_indices：图像节点对应的点云节点索引。
            # gt_pcd_node_corr_indices：点云节点对应的图像节点索引。
            # gt_img_node_corr_overlaps：图像节点和对应点云节点之间的重叠度。
            # gt_pcd_node_corr_overlaps：点云节点和对应图像节点之间的重叠度。

            gt_img_node_corr_indices += total_img_num_nodes
            gt_img_node_corr_levels = torch.full_like(gt_img_node_corr_indices, fill_value=i)
            all_gt_img_node_corr_levels.append(gt_img_node_corr_levels)
            all_gt_img_node_corr_indices.append(gt_img_node_corr_indices)
            all_gt_pcd_node_corr_indices.append(gt_pcd_node_corr_indices)
            all_gt_img_node_corr_overlaps.append(gt_img_node_corr_overlaps)
            all_gt_pcd_node_corr_overlaps.append(gt_pcd_node_corr_overlaps)
            # all_gt_img_node_corr_levels 存储了图像节点的层级信息，即每个节点对应的层级。
            # all_gt_img_node_corr_indices 存储了图像节点与点云节点之间的对应关系，即每个图像节点对应的点云节点的索引。
            # all_gt_pcd_node_corr_indices 存储了点云节点与图像节点之间的对应关系，即每个点云节点对应的图像节点的索引。
            # all_gt_img_node_corr_overlaps 存储了图像节点与点云节点之间的重叠度信息，即每个图像节点与其对应的点云节点之间的重叠度。
            # all_gt_pcd_node_corr_overlaps 存储了点云节点与图像节点之间的重叠度信息，即每个点云节点与其对应的图像节点之间的重叠度

            img_h_c //= 2
            img_w_c //= 2
            total_img_num_nodes += img_num_nodes

        img_node_masks = torch.cat(all_img_node_masks, dim=0)
        img_node_levels = torch.cat(all_img_node_levels, dim=0)

        output_dict["img_num_nodes"] = total_img_num_nodes#存储了图像节点的数量
        output_dict["pcd_num_nodes"] = pcd_points_c.shape[0]#存储了点云节点的数量

        gt_img_node_corr_levels = torch.cat(all_gt_img_node_corr_levels, dim=0)
        gt_img_node_corr_indices = torch.cat(all_gt_img_node_corr_indices, dim=0)
        gt_pcd_node_corr_indices = torch.cat(all_gt_pcd_node_corr_indices, dim=0)
        gt_img_node_corr_overlaps = torch.cat(all_gt_img_node_corr_overlaps, dim=0)
        gt_pcd_node_corr_overlaps = torch.cat(all_gt_pcd_node_corr_overlaps, dim=0)
        # gt_img_node_corr_levels 存储了所有图像节点对应的点云节点的层级信息。
        # gt_img_node_corr_indices 存储了所有图像节点对应的点云节点的索引信息。
        # gt_pcd_node_corr_indices 存储了所有点云节点对应的图像节点的索引信息。
        # gt_img_node_corr_overlaps 存储了所有图像节点对应的点云节点的重叠度信息。
        # gt_pcd_node_corr_overlaps 存储了所有点云节点对应的图像节点的重叠度信息。


        gt_node_corr_min_overlaps = torch.minimum(gt_img_node_corr_overlaps, gt_pcd_node_corr_overlaps)
        gt_node_corr_max_overlaps = torch.maximum(gt_img_node_corr_overlaps, gt_pcd_node_corr_overlaps)
        # gt_node_corr_min_overlaps 存储了图像节点和点云节点之间重叠度的最小值。这个张量的每个元素表示了对应的图像节点和点云节点之间的最小重叠度。
        # gt_node_corr_max_overlaps 存储了图像节点和点云节点之间重叠度的最大值。这个张量的每个元素表示了对应的图像节点和点云节点之间的最大重叠度。

        

        output_dict["gt_img_node_corr_indices"] = gt_img_node_corr_indices
        output_dict["gt_pcd_node_corr_indices"] = gt_pcd_node_corr_indices
        output_dict["gt_img_node_corr_overlaps"] = gt_img_node_corr_overlaps
        output_dict["gt_pcd_node_corr_overlaps"] = gt_pcd_node_corr_overlaps
        output_dict["gt_img_node_corr_levels"] = gt_img_node_corr_levels
        output_dict["gt_node_corr_min_overlaps"] = gt_node_corr_min_overlaps
        output_dict["gt_node_corr_max_overlaps"] = gt_node_corr_max_overlaps
        # output_dict["gt_img_node_corr_indices"] 存储了图像节点对应的点云节点的索引。
        # output_dict["gt_pcd_node_corr_indices"] 存储了点云节点对应的图像节点的索引。
        # output_dict["gt_img_node_corr_overlaps"] 存储了图像节点和点云节点之间的重叠度。
        # output_dict["gt_pcd_node_corr_overlaps"] 存储了点云节点和图像节点之间的重叠度。
        # output_dict["gt_img_node_corr_levels"] 存储了图像节点对应的点云节点的层级信息。
        # output_dict["gt_node_corr_min_overlaps"] 存储了图像节点和点云节点之间重叠度的最小值。
        # output_dict["gt_node_corr_max_overlaps"] 存储了图像节点和点云节点之间重叠度的最大值。
        
        # 5. Fine-leval matching
        img_channels_f = img_feats_f.shape[1]
        img_feats_f = img_feats_f.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
        pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)

        output_dict["img_feats_f"] = img_feats_f
        output_dict["pcd_feats_f"] = pcd_feats_f

        # 6. Select topk nearest node correspondences
        
        q1=pcd_feats_c

        #对图片特征进行均值方差采样/重参数化
        #ε ∼ N (0, I)高斯分布
        mean = torch.zeros(256).to(device)  # 均值为0
        covariance_matrix = torch.eye(256).to(device)  # 协方差矩阵为单位矩阵
        # 创建多元正态分布
        Gauss = MultivariateNormal(mean, covariance_matrix)

        all_img_feats_c0 = all_img_feats_c[0].view(24, 32, 256).permute(2, 0, 1)#将维度从[768,256]转变成了[256，24，32]
        all_img_feats_c1 = all_img_feats_c[1].view(12, 16, 256).permute(2, 0, 1)#将维度从[192,256]转变成了[256，12，16]
        all_img_feats_c2 = all_img_feats_c[2].view(6, 8, 256).permute(2, 0, 1)#将维度从[48,256]转变成了[256，6，8]

        #对all_img_feats_c0进行均值方差采样
        i0_mu1 = torch.mean(all_img_feats_c0, dim=[1, 2], keepdim=True)#做均值 #变为[1,256,24,32]
        i0_sig = self.i0_conv(all_img_feats_c0)#[1,256,1,1]
        i0_sig1 = i0_sig.squeeze(0)#[256,1,1]
        i0_mu2 = F.dropout(i0_mu1, p=0.5, training=True)
        i0_mu3 = i0_mu2.squeeze()#[256]
        i0_sig2 = i0_sig1.squeeze()#[256]
        i0_sig2s = F.softplus(i0_sig2)
        i0_sig2 = i0_sig2s + 1e-6

        d0 = i0_sig2.size(0)  # 获取方差向量的维度
        log_det_sigma0 = torch.sum(torch.log(i0_sig2))  # 计算对角协方差矩阵的行列式的对数
        entropy0 = 0.5 * (d0 * torch.log(torch.tensor(2 * torch.pi)) + log_det_sigma0)
        output_dict["entropy0"] = entropy0

        i0_sig2sqrt =  torch.sqrt(i0_sig2).to(device)
        all_img_feats_c_0_sample = Gauss.sample((768,))  # 生成768个样本
        all_img_feats_c_0 = i0_mu3 + all_img_feats_c_0_sample*i0_sig2sqrt

        #对all_img_feats_c1进行均值方差采样
        i1_mu1 = torch.mean(all_img_feats_c1, dim=[1, 2], keepdim=True)#做均值 #变为[1,256,12,16]
        i1_sig = self.i1_conv(all_img_feats_c1)#[1,256,1,1]
        i1_sig1 = i1_sig.squeeze(0)#[256,1,1]
        i1_mu2 = F.dropout(i1_mu1, p=0.5, training=True)
        i1_mu3 = i1_mu2.squeeze()#[256]
        i1_sig2 = i1_sig1.squeeze()#[256]
        i1_sig2s = F.softplus(i1_sig2)
        i1_sig2 = i1_sig2s + 1e-6

        d1 = i1_sig2.size(0)  # 获取方差向量的维度
        log_det_sigma1 = torch.sum(torch.log(i1_sig2))  # 计算对角协方差矩阵的行列式的对数
        entropy1 = 0.5 * (d1 * torch.log(torch.tensor(2 * torch.pi )) + log_det_sigma1)
        output_dict["entropy1"] = entropy1
  
        i1_sig2sqrt =  torch.sqrt(i1_sig2).to(device)
        all_img_feats_c_1_sample = Gauss.sample((192,))  # 生成192个样本
        all_img_feats_c_1 = i1_mu3 + all_img_feats_c_1_sample*i1_sig2sqrt
        
        
        q1 = q1.unsqueeze(0)

        i2,q2 = self.crosstransformer(
            all_img_feats_c_0.unsqueeze(0),
            img_pixels_c.unsqueeze(0),
            q1,
            pcd_points_c.unsqueeze(0)
            )
        i3,q3 = self.crosstransformer(
            all_img_feats_c_1.unsqueeze(0),
            img_pixels_c.unsqueeze(0),
            q2,
            pcd_points_c.unsqueeze(0)       
            )

        q1 = q1.squeeze(0)
        q2 = q2.squeeze(0)   
        q3 = q3.squeeze(0)
        
        q1 = F.normalize(q1, p=2, dim=1)
        q2 = F.normalize(q2, p=2, dim=1)
        q3 = F.normalize(q3, p=2, dim=1)
        
        output_dict["q1"] = q1
        output_dict["q2"] = q2
        output_dict["q3"] = q3
        output_dict["img_node_masks"] = img_node_masks
        output_dict["pcd_node_masks"] = pcd_node_masks

        output_dict["all_img_feats_c[0]"] = all_img_feats_c[0]
        output_dict["all_img_feats_c[1]"] = all_img_feats_c[1]
        output_dict["all_img_feats_c[2]"] = all_img_feats_c[2]

        if not self.training:
            img_node_corr_indices, pcd_node_corr_indices, node_corr_scores = self.coarse_matching(
                img_feats_c, q1,q2,q3, img_node_masks, pcd_node_masks)

            img_node_corr_levels = img_node_levels[img_node_corr_indices]
            output_dict["img_node_corr_indices"] = img_node_corr_indices
            output_dict["pcd_node_corr_indices"] = pcd_node_corr_indices
            output_dict["img_node_corr_levels"] = img_node_corr_levels

            pcd_padded_feats_f = torch.cat([pcd_feats_f, torch.zeros_like(pcd_feats_f[:1])], dim=0)#通过在点云特征 pcd_feats_f 的末尾添加一行全零的特征向量来填充点云特征。

            # 7. Extract patch correspondences
            all_img_corr_indices = []
            all_pcd_corr_indices = []

            for i in range(self.img_num_levels_c):
                node_corr_masks = torch.eq(img_node_corr_levels, i)

                if node_corr_masks.sum().item() == 0:
                    continue

                cur_img_node_corr_indices = img_node_corr_indices[node_corr_masks] - all_img_total_nodes[i]
                cur_pcd_node_corr_indices = pcd_node_corr_indices[node_corr_masks]

                img_node_knn_points = all_img_node_knn_points[i]
                img_node_knn_pixels = all_img_node_knn_pixels[i]
                img_node_knn_indices = all_img_node_knn_indices[i]

                img_node_corr_knn_indices = index_select(img_node_knn_indices, cur_img_node_corr_indices, dim=0)
                img_node_corr_knn_masks = torch.ones_like(img_node_corr_knn_indices, dtype=torch.bool)
                img_node_corr_knn_feats = index_select(img_feats_f, img_node_corr_knn_indices, dim=0)

                pcd_node_corr_knn_indices = pcd_node_knn_indices[cur_pcd_node_corr_indices]  # (P, Kc)
                pcd_node_corr_knn_masks = pcd_node_knn_masks[cur_pcd_node_corr_indices]  # (P, Kc)
                pcd_node_corr_knn_feats = index_select(pcd_padded_feats_f, pcd_node_corr_knn_indices, dim=0)

                similarity_mat = pairwise_cosine_similarity(
                    img_node_corr_knn_feats, pcd_node_corr_knn_feats, normalized=True
                )

                batch_indices, row_indices, col_indices, _ = batch_mutual_topk_select(
                    similarity_mat,
                    k=1,
                    row_masks=img_node_corr_knn_masks,
                    col_masks=pcd_node_corr_knn_masks,
                    threshold=0.75,
                    largest=True,
                    mutual=True,
                )

                img_corr_indices = img_node_corr_knn_indices[batch_indices, row_indices]
                pcd_corr_indices = pcd_node_corr_knn_indices[batch_indices, col_indices]

                all_img_corr_indices.append(img_corr_indices)
                all_pcd_corr_indices.append(pcd_corr_indices)

            img_corr_indices = torch.cat(all_img_corr_indices, dim=0)
            pcd_corr_indices = torch.cat(all_pcd_corr_indices, dim=0)

            # duplicate removal
            num_points_f = pcd_points_f.shape[0]
            corr_indices = img_corr_indices * num_points_f + pcd_corr_indices
            unique_corr_indices = torch.unique(corr_indices)
            img_corr_indices = torch.div(unique_corr_indices, num_points_f, rounding_mode="floor")
            pcd_corr_indices = unique_corr_indices % num_points_f

            img_points_f = img_points_f.view(-1, 3)
            img_pixels_f = img_pixels_f.view(-1, 2)
            img_corr_points = img_points_f[img_corr_indices]
            img_corr_pixels = img_pixels_f[img_corr_indices]
            pcd_corr_points = pcd_points_f[pcd_corr_indices]
            pcd_corr_pixels = pcd_pixels_f[pcd_corr_indices]
            img_corr_feats = img_feats_f[img_corr_indices]
            pcd_corr_feats = pcd_feats_f[pcd_corr_indices]
            corr_scores = (img_corr_feats * pcd_corr_feats).sum(1)

            output_dict["img_corr_points"] = img_corr_points
            output_dict["img_corr_pixels"] = img_corr_pixels
            output_dict["img_corr_indices"] = img_corr_indices
            output_dict["pcd_corr_points"] = pcd_corr_points
            output_dict["pcd_corr_pixels"] = pcd_corr_pixels
            output_dict["pcd_corr_indices"] = pcd_corr_indices
            output_dict["corr_scores"] = corr_scores

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration

        return output_dict


def create_model(cfg):
    model = B23D(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    main()
