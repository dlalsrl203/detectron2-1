# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import Backbone, build_backbone
from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = [
    "SemanticSegmentor",
    "SEM_SEG_HEADS_REGISTRY",
    "SemSegFPNHead",
    "build_sem_seg_head",
]


SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


@META_ARCH_REGISTRY.register()
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.
    ##
    :paper:`PanopticFPN`에 설명된 시맨틱 분할 헤드.
    FPN features 목록을 입력으로 사용하고 3x3 전환 및 업샘플링 시퀀스를 적용하여 모든 항목을 ``common_stride``로 정의된 보폭으로 확장합니다.
    그런 다음 이러한 features가 추가되어 다른 1x1 변환 레이어에서 최종 예측을 수행하는 데 사용됩니다.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
            ##
            input_shape: 입력 features의 모양(채널 및 보폭)
            num_classes: 예측할 클래스 수
            conv_dims: 중간 변환 레이어의 출력 채널 수입니다.
            common_stride: 모든 features이 다음으로 업그레이드될 것이라는 공통된 진전
            loss_weight: loss weight
            norm (str or callable): 모든 전환 레이어에 대한 정규화
            ignore_value: 학습 중에 무시할 카테고리 ID입니다.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride) # stride 순으로 정렬
        if not len(input_shape):
            raise ValueError("SemSegFPNHead(input_shape=) cannot be empty!")
        self.in_features = [k for k, v in input_shape] # 정렬된 순으로 feature 저장
        feature_strides = [v.stride for k, v in input_shape] # fzip(["p2", "p3", "p4", "p5"], [4, 8, 16, 32]) // eature의 stride 저장
        # feature = 0, 1, 2, 3
        feature_channels = [v.channels for k, v in input_shape] # feature의 channel 저장

        self.ignore_value = ignore_value # 255 // 학습 중에 무시할 카테고리 ID
        self.common_stride = common_stride # 4 // 모든 features가 upscale될 기본 stride
        self.loss_weight = loss_weight # 0.5

        self.scale_heads = [] # feature의 크기와 채널을 변경시키는 head
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops = [] # Convolution + Nomalize + Upsample 3개 묶음
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                # convolution layer 생성.
                conv = Conv2d( 
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3, # 3x3
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module, # Nomalize 추가
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv) # Caffe2에 구현된 "MSRAFill"을 사용하여 `module.weight`를 초기화합니다.   
                                               # 또한 `module.bias`를 0으로 초기화합니다.
                head_ops.append(conv) # 생성한 conv 추가
                if stride != self.common_stride: # p2가 아니면이라는 건가?
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    ) # upsample 추가
                    '''
                        nn.Upsample(size, scale_factor, mode, align_corners)
                            size = 출력 공간 크기
                            scale_factor = 공간 크기에 대한 승수. 튜플인 경우 입력 크기와 일치해야 합니다.
                            mode = 업샘플링 알고리즘(보간법): 'nearest' , 'linear' , 'bilinear' , 'bicubic' 및 'trilinear' 중 하나입니다. 기본값: 'nearest'
                            align_corners = True 이면 입력 및 출력 텐서의 모서리 픽셀이 정렬되어 해당 픽셀의 값을 유지합니다.
                                             mode 가 'linear' , 'bilinear' 또는 'trilinear' 인 경우에만 효과가 있습니다 . 기본값 : False
                    '''
            self.scale_heads.append(nn.Sequential(*head_ops)) # nn.Sequential 신경망 생성 https://dororongju.tistory.com/147
            self.add_module(in_feature, self.scale_heads[-1]) # scale_heads에 추가된 제일 마지막 걸 추가
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0) # 마지막 예측층
        '''
        conv_dims = 4
        num_classes = 54
        '''
        weight_init.c2_msra_fill(self.predictor) # Caffe2에 구현된 "MSRAFill"을 사용하여 `module.weight`를 초기화합니다.  

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
        }

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
