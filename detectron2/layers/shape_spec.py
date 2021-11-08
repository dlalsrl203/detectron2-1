# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    ##
    텐서의 기본 모양 사양을 포함하는 간단한 구조입니다.
    pytorch 모듈 간의 모양 추론 기능 부족을 보완하기 위해 모델의 보조 입력/출력으로 자주 사용됩니다.

    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)
