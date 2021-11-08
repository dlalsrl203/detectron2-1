# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

__all__ = [
    "BaseMaskRCNNHead",
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "ROI_MASK_HEAD_REGISTRY",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    ##
    pred_mask_logits를 예측된 전경 확률 마스크로 변환하는 동시에 pred_instances에서 예측된 클래스에 대한 마스크만 추출합니다. 
    각 예측 상자에 대해 pred_instances에 새로운 "pred_masks" 필드를 추가하여 동일한 클래스의 마스크가 인스턴스에 첨부됩니다.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
        ##
        pred_mask_logits (Tensor): 
            클래스 특정 또는 클래스 불가지론에 대한 (B, C, Hmask, Wmask) 또는 (B, 1, Hmask, Wmask) 모양의 텐서. 
            여기서 B는 모든 이미지에서 예측된 마스크의 총 수이고, C는 다음의 수입니다. 
            전경 클래스 및 Hmask, Wmask는 마스크 예측의 높이와 너비입니다. 값은 로짓입니다.
        pred_instances (list[Instances]): 
            N 인스턴스의 목록입니다. 여기서 N은 배치의 이미지 수입니다. 
            각 인스턴스에는 "pred_classes" 필드가 있어야 합니다.

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
        ##
        없음. 
            pred_instances는 예측된 클래스에 대한 크기(Hmask, Wmask)의 마스크를 저장하는 추가 "pred_masks" 필드를 포함합니다. 
            마스크는 네트워크에서 예측한 해상도의 소프트(양자화되지 않은) 마스크로 반환됩니다. 
            예측된 마스크의 크기를 원래 이미지 해상도로 조정하거나 이진화하는 것과 같은 후처리 단계는 호출자에게 맡겨집니다.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        # 예측된 클래스에 해당하는 마스크 선택
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0):
        """
        NOTE: this interface is experimental.

        Args:
            loss_weight (float): multiplier of the loss
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
'''
토치스크립트 지원을 받기 위해 헤드를 `nn.Sequential`의 하위 클래스로 만듭니다.
따라서 이 헤드 클래스에 새 레이어를 추가하려면 forward()에서 사용할 순서대로 추가해야 합니다.
'''
@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    ##
    여러 변환 레이어와 업샘플 레이어('ConvTranspose2d' 포함)가 있는 마스크 헤드.
    예측은 최종 1x1 전환 레이어로 이루어집니다.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.
        ##
        이 인터페이스는 실험적입니다.

        Args:
            input_shape (ShapeSpec): shape of the input feature 
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            ##
            input_shape (ShapeSpec): 입력 feature의 모양
            num_classes (int): 전경 클래스의 수(즉, 배경은 포함되지 않음).
             클래스 불가지론적(몇몇 명제의 진위여부를 알 수 없다고 보는 철학적 관점) 예측을 사용하는 경우 1입니다.
            conv_dims (list[int]): N-1 변환 레이어와 마지막 업샘플 레이어의 출력 차원을 나타내는 N>0 정수 목록입니다.
            conv_norm (str or callable): 변환 레이어에 대한 정규화.
                지원되는 유형은 :func:`detectron2.layers.get_norm`을 참조하세요.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,   #input_shape.channels
                conv_dim,
                kernel_size=3,  # Filter 3x3
                stride=1,
                padding=1,      # 0 padding
                bias=not conv_norm, # conv_norm 문자열이면 flase?
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),   # 활성화 함수 ReLU
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        # sample model. It has nn.ConvTranspose2d(1, 3, 4, 1, 0, bias = False)
        # First parameter = input의 채널수 (=1)
        # Second parameter = output의 채널 수 (=3)
        # Third parameter = Kernel size (=4)
        # Fourth parameter = stride (=1)
        # fifth parameter = padding (=0)
        ##
        # ConvTranspose2d 
        # 이 모듈은 입력에 대한 Conv2d의 기울기로 볼 수 있습니다. 
        # 이것은 fractionally-strided convolution 또는 deconvolution으로도 알려져 있습니다.
        # (컨볼루션의 진정한 역을 계산하지 않기 때문에 실제 deconvolution 연산은 아니지만).
        # deconvolution = upsampling 
        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer) # Caffe2에 구현된 "MSRAFill"을 사용하여 `module.weight`를 초기화합니다.
                                            # 또한 `module.bias`를 0으로 초기화합니다.
        # use normal distribution initialization for mask prediction layer
        # .normal_(tensor, mean=0.0, std=1.0)
        # 마스크 예측 레이어에 정규 분포 초기화 사용 (정규 분포 N(mean, std^2)에서 가져온 값으로 입력 Tensor를 채웁니다.)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            # val 값으로 입력 Tensor를 채웁니다.
            # nn.init.constant_(tensor, val)
            nn.init.constant_(self.predictor.bias, 0) # bias를 0으로 채운다.

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
