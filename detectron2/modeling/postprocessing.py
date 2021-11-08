# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F

from detectron2.structures import Instances, ROIMasks


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    ##
    출력 인스턴스의 크기를 조정합니다.
    입력 이미지는 객체 감지기에 들어갈 때 종종 크기가 조정됩니다.
    결과적으로 우리는 종종 입력과 다른 해상도의 감지기 출력이 필요합니다.

    이 기능은 R-CNN 검출기의 원시 출력 크기를 조정하여 원하는 출력 해상도에 따라 출력을 생성합니다.
    
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
        ##
        results (Instances): 검출기의 원시 출력.
            `results.image_size`는 감지기가 보는 입력 이미지 해상도를 포함합니다.
            이 개체는 내부에서 수정될 수 있습니다.
        output_height, output_width: 원하는 출력 해상도.


    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor): # output_height가 torch.Tensor 타입의 자료형인지 확인
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        ##
        # scale_x 및 scale_y를 계산할 때 실제 나누기가 수행되도록 정수 텐서를 부동 임시로 변환합니다.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())
    ## result가 bounding bax라면
    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"
    # 객체 사이즈를 변경하고 이미지 사이즈에 맞게 자른다.
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    # result가 mask 타입이라면
    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks): # results.pred_masks가 ROIMasks 타입인지
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :]) #(N, 1, M, M) => (N, 0, M, M)
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"): # result가 keypoint 타입이라면
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.
    ##
    원래 해상도로 semantic segmentation predictions을 반환합니다.
    semantic segmentor를 입력할 때 입력 이미지의 크기가 조정되는 경우가 많습니다. 또한 동일한 경우 최대 네트워크 보폭으로 나눌 수 있도록 세그먼트 내부도 패딩했습니다.
    결과적으로 입력과 다른 해상도의 세그먼트 예측이 필요한 경우가 많습니다.
    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.
        ##
        result (Tensor): 시맨틱 분할 예측 로짓. (C, H, W) 모양의 텐서.
         여기서 C는 클래스 수이고 H, W는 예측의 높이와 너비입니다.
        img_size (tuple): 세그먼트가 입력으로 사용하는 이미지 크기입니다.
        output_height, output_width: 원하는 출력 해상도

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
        ##
        semantic segmentation prediction (Tensor): 픽셀당 소프트 예측을 포함하는 모양(C, output_height, output_width)의 텐서입니다.

    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    # F.interpolate은 작은 feature의 크기를 크게 변경시킬때 사용된다.
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0] 
    return result
