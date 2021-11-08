# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from torchvision.ops import roi_align


# NOTE: torchvision's RoIAlign has a different default aligned=False
# 토치비전의 RoIAlign에 다른 기본값이 정렬됨=False가 있습니다.
class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.
            ##
            output_size (tuple): h, w
            spatial_scale (float): 이 숫자로 입력 상자의 크기를 조정합니다.
            sampling_ratio (int): 각 출력 샘플에 대해 취할 입력 샘플의 수. 0은 샘플을 조밀하게 취합니다.
            aligned (bool): False인 경우 Detectron의 레거시 구현을 사용합니다. True이면 결과를 더 완벽하게 정렬합니다.
             = False 일 경우, 기존의 RoIAlign을 사용하고, True일 경우에만 RoIAlign V2를 사용한다.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.
            ##
            연속 좌표 c가 주어지면 두 개의 인접 픽셀 인덱스(우리 픽셀 모델에서)는 floor(c - 0.5) 및 ceil(c - 0.5)에 의해 계산됩니다. 
            예를 들어, c=1.3에는 이산 인덱스 [0] 및 [1](연속 좌표 0.5 및 1.5의 기본 신호에서 샘플링됨)이 있는 픽셀 이웃이 있습니다. 
            그러나 원래 roi_align(aligned=False)은 인접 픽셀 인덱스를 계산할 때 0.5를 빼지 않으므로 쌍선형 보간을 수행할 때 
            약간 잘못된 정렬(픽셀 모델에 비해)이 있는 픽셀을 사용합니다.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.
            ##
            먼저 ROI를 적절하게 조정한 다음 roi_align을 호출하기 전에 -0.5만큼 이동합니다. 
            이것은 올바른 neighbors을 생성합니다. 확인을 위해 detectron2/tests/test_roi_align.py를 참조하십시오.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
            ##
            ROIAlign이 변환 레이어와 함께 사용되는 경우 차이는 모델의 성능에 차이를 만들지 않습니다.
        """
        super().__init__()
        self.output_size = output_size # h, w
        self.spatial_scale = spatial_scale # 이 숫자로 입력 상자의 크기를 조정합니다.
        self.sampling_ratio = sampling_ratio # 각 출력 샘플에 대해 취할 입력 샘플의 수. 0은 샘플을 조밀하게 취합니다.
        self.aligned = aligned # False 일 경우, 기존의 RoIAlign을 사용하고, True일 경우에만 RoIAlign V2(-0.5)를 사용한다.

        from torchvision import __version__

        version = tuple(int(x) for x in __version__.split(".")[:2])
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
                Bx5 상자. 첫 번째 열은 N에 대한 인덱스입니다. 다른 4개 열은 xyxy입니다.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if input.is_quantized: # Tensor가 양자화되면 True이고 그렇지 않으면 False입니다.
            input = input.dequantize()
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
