# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


# config setup
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # panoptic-DeepLab의 데모를 사용하려면 밑에 두줄의 주석을 제거
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # 설정 동결
    cfg.freeze()
    return cfg

# 사용자로부터 설정 값을 입력받는다.
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

# codec과 file_ext로 video 생성이 되는지 테스트
def test_opencv_video_format(codec, file_ext):
    # 임시 폴더 생성
    # with : 자원을 사용하고 반납할때 사용
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        # 테스트 파일 생성 성공
        if os.path.isfile(filename):
            return True
        # 테스트 파일 생성 실패
        return False

# 같은 파일 내에서 호출되면 __name__ = __main__ 다른 파일에서 호출되면 파일 명 반환
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # 사용자로부터 args를 입력 받는다.
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)# 사용자로부터 입력받은 값들로 설정

    # VisualizationDemo 초기화(시각화)
    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            # 사용자의 홈 디렉터리
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            # 에러가 발생할 경우 에러메시지 출력
            assert args.input, "The input path(s) was not found"
            
            # tqdm (프로세스바)
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            # 평가와 일치하도록 PIL 사용
            # PIL : python image library
            img = read_image(path, format="BGR") # 1. Read
            start_time = time.time()
            # predictions : 모델의 출력
            # visualized_output : 시각화된 이미지 출력
            predictions, visualized_output = demo.run_on_image(img)
            # 진행상황 log로 출력
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                # 사용자로부터 입력받은 결과 출력 디렉터리가 존재하는지 확인 (isdir 존재 여부 확인)
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    #output과 경로의 기본이름(basename)을 연결한다.
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                # 시각화된 이미지를 저장한다. 
                visualized_output.save(out_filename)
            else:
                # WINDOW_NAME이라는 이름의 창을 생성, cv2.namedWindow(winname, flags)
                # flags 
                # 1. cv2.WINDOW_NORMAL : 사용자가 창 크기를 조정할 수 있음.
                # 2. cv2.WINDOW_AUTOSIZE : 이미지와 동일한 크기로 창 크기를 재조정할 수 있음.
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # 이미지를 모니터에 출력 
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                # esc가 입력되는 것을 기다린다. 0은 무한 대기, 27은 esc
                # esc 입력 시 종료
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
        # webcam 사용
    elif args.webcam:
        # input이나 output이 있다면 에러메시지 출력
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        # 0번 카메라
        cam = cv2.VideoCapture(0)
        # 프로세스바 출력
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # 0번 카메라를 화면에 출력
            cv2.imshow(WINDOW_NAME, vis)
            # cv2.waitKey(1) : 1ms 동안 기다리고 화면을 새로고침하여 동영상을 출력한다.
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        # release() : webcam 사용 종료
        cam.release()
        # cv2.destroyAllWindows() : 열려있는 모든 창을 닫는다.
        cv2.destroyAllWindows()
        # video input
    elif args.video_input:
        # args.video_input 경로에 있는 video 파일을 읽어온다.
        video = cv2.VideoCapture(args.video_input)
        # width : 프레임 폭
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height : 프레임 높이
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frames_per_second : 초당 프레임 수(fps)
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        # num_frames : 프레임 수
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # basename : args.video_input의 기본 파일 명
        basename = os.path.basename(args.video_input)
        # 코덱과 확장자
        codec, file_ext = (
            # 테스트 파일이 생성되는지 확인
            # ("x264", ".mkv")으로 테스트 파일이 생성 안되면 ("mp4v", ".mp4")으로 변경
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        # 코덱이 mp4v라면 x264를 사용할 수 없음
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                # os.path.splitext 파일 명을 파일 명과 확장자로 나눈다.
                # 파일명에 확장자를 합친다.
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
                # 같은 이름의 파일이 없으면 output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            # video 파일 생성
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                # cv2.VideoWriter_fourcc : 코덱 정보
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        # 프로세스 바 출력
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                # 동영상이기 때문에 화면을 새로고침하기 위해서 1로 설정(1000이 1초) 0이면 새로고침 안됨
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
