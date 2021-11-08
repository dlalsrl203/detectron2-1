# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import math
from typing import List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.registry import Registry

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = """
Registry for modules that creates object detection anchors for feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
"""


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.
    ##
    하나의 크기(또는 종횡비)가 지정되고 여러 feature 맵이 있는 경우 모든 feature 맵에 대해 해당 단일 크기(또는 종횡비)의 앵커를 "브로드캐스트"합니다.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.
    ##
    params가 list[float]이거나 len(params) == 1인 list[list[float]]인 경우 num_features 시간을 반복합니다.
    Returns:
        list[list[float]]: param for each feature
        ##
        각 feature에 대한 매개변수
    """
    assert isinstance(
        params, collections.abc.Sequence
    ), f"{name} in anchor generator has to be a list! Got {params}."
    # collections.abc.Sequence : 추상 클래스가 요구하는 메서드를 모두 구현하면 별도로 작업하지 않아도 클래스가 index와 count 같은 부가적인 메서드들 모두 제공
    
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params


@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    ##
    "Faster R-CNN: 지역 제안 네트워크를 통한 실시간 객체 감지를 향하여"에 설명된 표준 방식으로 앵커를 계산합니다.
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    ##
    각 앵커 박스의 치수(차원).
    """

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.
        ##
        이 인터페이스는 실험적입니다.

        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
                ##
                ``sizes``가 list[list[float]]인 경우 ``sizes[i]``는 i번째 피쳐 맵에 사용할 앵커 크기(즉, 앵커 영역의 sqrt) 목록입니다.
                ``sizes``가 list[float]이면 모든 기능 맵에 ``sizes``가 사용됩니다.
                앵커 크기는 입력 이미지 단위의 절대 길이로 제공됩니다. 입력 이미지 크기가 변경되면 동적으로 크기가 조정되지 않습니다.

            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
                ##
                앵커에 사용할 종횡비(즉, 높이/너비) 목록입니다. '크기'에 대해 동일한 "브로드캐스트" 규칙이 적용됩니다.

            strides (list[int]): stride of each input feature.
                ##
                각 input feature의 스트라이드.

            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
                ##
                첫 번째 앵커의 중심과 이미지의 왼쪽 위 모서리 사이의 상대적 오프셋입니다. 값은 [0, 1)에 있어야 합니다.
                반 보폭을 의미하는 0.5를 사용하는 것이 좋습니다.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset
        '''
        assert는 이 조건이 참일때 코드는 내가 보장한다. 이 조건은 올바르다!
        하지만 이 조건이 거짓이라는 것은 내가 보증하지 않은 동작이다. 그러니 AssertionError를 발생해라.
        '''

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        '''
        zip : zip(sizes, aspect_ratios) = [(sizes[0], aspect_radios[0]), (sizes[1], aspect_radios[1]) ~ (sizes[n], aspect_radios[n])]
        '''
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        `num_anchors`의 별칭
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
                ##
                각 int는 해당 기능 맵의 모든 픽셀 위치에 있는 앵커 수입니다.
                예를 들어 모든 픽셀에서 3개의 종횡비와 5개의 크기의 앵커를 사용하는 경우 앵커의 수는 15입니다.
                (구성에서 ANCHOR_GENERATOR.SIZES 및 ANCHOR_GENERATOR.ASPECT_RATIOS 참조)

                표준 RPN 모델에서 모든 기능 맵의 'num_anchors'는 동일합니다.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
            ##
            featuremap 텐서, 각각은 (#locations x #cell_anchors) x 4입니다.
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        # 버퍼()는 토치스크립트에서 지원하지 않습니다. 대신 named_buffers() 사용
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
            # reshape(-1, 4)) : -1은 모든 원소가 빠짐 없이 배치될 수 있도록 열이 가변적으로 변한다. 
            '''
            reshape(-1, 4)의 예시
            array(
                [[ 0, 1, 2, 3],
                [ 4, 5, 6, 7],
                [ 8, 9, 10, 11]]
            )
            '''

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        ##
        (0, 0)에 중심을 둔 다양한 크기와 종횡비의 모든 앵커 박스인 표준 앵커 박스를 저장하는 텐서를 생성합니다.
        나중에 이 텐서를 이동하고 타일링하여 전체 기능 맵에 대한 앵커 세트를 빌드할 수 있습니다(`meth:_grid_anchors` 참조).
        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
            모양의 텐서(len(sizes) * len(aspect_ratios), 4) 앵커 상자를 XYXY 형식으로 저장합니다.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227
        '''
        이것은 원래 Faster R-CNN 코드 또는 Detectron에 정의된 앵커 생성기와 다릅니다. 
        그것들은 동일한 AP를 산출하지만, 이전 버전은 다른 종횡비에 대해 약간 다른 크기를 초래하는 피쳐 그리드 및 양자화에 대한 
        상대적인 이동으로 덜 자연스러운 방식으로 셀 앵커를 정의합니다.
        '''

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                #sqrt = 제곱근(루트)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.
            ##
            앵커를 생성할 백본 feature map 목록입니다.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
            ##
            각 feature map에 대한 모든 앵커를 포함하는 상자 목록(즉, feature map의 모든 위치에서 반복되는 셀 앵커).
            각 feature map의 앵커 수는 Hi x Wi x num_cell_anchors이며, 여기서 Hi, Wi는 특징 맵의 해상도를 앵커 스트라이드로 나눈 값입니다.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]


@ANCHOR_GENERATOR_REGISTRY.register()
class RotatedAnchorGenerator(nn.Module):
    """
    Compute rotated anchors used by Rotated RPN (RRPN), described in
    "Arbitrary-Oriented Scene Text Detection via Rotation Proposals".
    """

    box_dim: int = 5
    """
    the dimension of each anchor box.
    """

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, angles, offset=0.5):
        """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If sizes is list[list[float]], sizes[i] is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If sizes is list[float], the sizes are used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            angles (list[list[float]] or list[float]): list of angles (in degrees CCW)
                to use for anchors. Same "broadcast" rule for `sizes` applies.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        angles = _broadcast_params(angles, self.num_features, "angles")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios, angles)

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
            "angles": cfg.MODEL.ANCHOR_GENERATOR.ANGLES,
        }

    def _calculate_anchors(self, sizes, aspect_ratios, angles):
        cell_anchors = [
            self.generate_cell_anchors(size, aspect_ratio, angle).float()
            for size, aspect_ratio, angle in zip(sizes, aspect_ratios, angles)
        ]
        return BufferList(cell_anchors)

    @property
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios, 2 sizes and 5 angles, the number of anchors is 30.
                (See also ANCHOR_GENERATOR.SIZES, ANCHOR_GENERATOR.ASPECT_RATIOS
                and ANCHOR_GENERATOR.ANGLES in config)

                In standard RRPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            zeros = torch.zeros_like(shift_x)
            shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)

            anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5))

        return anchors

    def generate_cell_anchors(
        self,
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1, 2),
        angles=(-90, -60, -30, 0, 30, 60, 90),
    ):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes, aspect_ratios, angles centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):
            angles (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios) * len(angles), 5)
                storing anchor boxes in (x_ctr, y_ctr, w, h, angle) format.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                anchors.extend([0, 0, w, h, a] for a in angles)

        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[RotatedBoxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [RotatedBoxes(x) for x in anchors_over_all_feature_maps]


def build_anchor_generator(cfg, input_shape):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)
