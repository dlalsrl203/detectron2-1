# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode): ColorMode.IMAGE - 모든 인스턴스에 대해 임의의 색상을 선택하고 불투명도가
                                                         낮은 오버레이 분할을 선택합니다.
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
                시각화와 다른 프로세스에서 모델을 실행할지 여부.
                시각화 논리가 느릴 수 있으므로 유용합니다.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        # 모델을 cpu에서 부른다.
        # device 선언(cpu)
        self.cpu_device = torch.device("cpu")
        # color mode 선언
        # IMAGE -> Random Color
        # SEGMENTATION -> 같은 instance는 비슷한 색상으로 처리
        # IMAGE_BW -> masks 들을 제외한 모든 영역을 gray-scale 처리(회색)
        self.instance_mode = instance_mode
        # GPU 병렬 처리 (False이므로 사용 안함)
        self.parallel = parallel
        # 다른 프로세스에서 모델을 실행(gpu 활용)
        if parallel:
            # gpu 활용
            # 사용가능한 gpu 갯수 반환
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            # gpu 활용 x
            self.predictor = DefaultPredictor(cfg)


    # 이미지 데이터를 예측하고 시각화
    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
                형상(H, W, C)의 이미지(BGR 순서).
                OpenCV에서 사용하는 형식입니다.
        Returns:
            predictions (dict): the output of the model.
                                모델의 출력.
            vis_output (VisImage): the visualized image output.
                                    시각화된 이미지 출력
        """
        vis_output = None
        # AsyncPredictor(gpu) 또는 DefaultPredictor(cpu)에 이미지를 보내고 예측
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # ::-1  =  처음부터 끝까지 역순으로
        # BGR => RGB
        image = image[:, :, ::-1]
        #시각화 도우미
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        
        # panoptic segmentation인지 확인
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            # semantic segmentation인지 확인
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            # instances segmentation인지 확인
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    # 비디오를 열고 프레임을 반환한다.
    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    # 비디오 데이터를 예측하고 시각화한다.
    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        ##
        입력 비디오의 프레임에 대한 예측을 시각화합니다.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
                `VideoCapture` 객체. 소스는 웹캠 또는 비디오 파일일 수 있습니다.

        Yields:
            ndarray: BGR visualizations of each video frame.
                    각 비디오 프레임의 BGR 시각화.
        """
        # 비디오 데이터 시각화 도우미
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            #cv2.cvtColor frame을 bgr에서 rgb로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # panotic segmentation 시각화
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            # instance segmentation 시각화
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            # semantic segmentation 시각화
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            # Matplotlib RGB 형식을 OpenCV BGR 형식으로 변환
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        # 비디오 프레임
        frame_gen = self._frame_from_video(video)
        
        # gpu 사용여부
        if self.parallel:
            # gpu 사용
            # AsyncPredictor의 기본 buffer size
            buffer_size = self.predictor.default_buffer_size
            
            # double-ended queue : 양방향에서 데이터 처리 가능한 큐 (덱)
            frame_data = deque()
            
            # enumerate : 몇 번째 반복문인지 확인
            for cnt, frame in enumerate(frame_gen):
                # 큐에 frame 추가
                frame_data.append(frame)
                # predictor에 frame 추가
                self.predictor.put(frame)

                # 프레임 수가 버퍼 사이즈를 초과하면
                if cnt >= buffer_size:
                    # popleft() : 큐의 가장 왼쪽에 있는 원소를 제거하고 그 값을 리턴
                    frame = frame_data.popleft()
                    # 모델이 예측한 결과를 반환 받는다. 
                    predictions = self.predictor.get()
                    # 시각화한 결과를 반환
                    yield process_predictions(frame, predictions)

            # 프레임 수가 버퍼 사이즈를 초과하지 않았을 경우
            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                # 시각화한 결과를 반환
                yield process_predictions(frame, predictions)
        # gpu 사용 x
        else:
            for frame in frame_gen:
                # 시각화한 결과를 반환
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            # gpu로 설정된 값으로 predictor 생성
            predictor = DefaultPredictor(self.cfg)

            while True:
                # 작업할 queue
                task = self.task_queue.get()
                # task가 _StopToken 클래스이면 true
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                # 결과 큐에 저장
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        # 사용할 gpu 수
        num_workers = max(num_gpus, 1)
        # 멀티 프로세싱(스레드)로 작업할 작업리스트 큐 생성 gpu * 3
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        # 멀티 프로세싱(스레드)로 작업한 작업 결과 리스트 큐 생성 gpu * 3
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        #
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            # 설정 동결 해제 (cfg를 수정하겠다.)
            cfg.defrost()
            # gpu가 있으면 cuda:{}".format(gpuid) 없으면 cpu(기본은 cpu)
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        # 프로그램 종료(AsyncPredictor 종료)
        atexit.register(self.shutdown)

    def put(self, image):
        # 인덱스 1 증가
        self.put_idx += 1
        # 작업 큐에 인덱스와 이미지를 넣는다.
        self.task_queue.put((self.put_idx, image))

    def get(self):
        # self.get_idx : default = 0
        self.get_idx += 1  # the index needed for this request
                            # 이 요청에 필요한 인덱스
        # self.result_rank : default = []
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            # self.result_data[0], self.result_rank[0] 삭제
            del self.result_data[0], self.result_rank[0]
            # 삭제되기 전 self.result_data[0] 반환
            return res

        while True:
            # make sure the results are returned in the correct order
            # 결과가 올바른 순서로 반환되었는지 확인하십시오.
            # 모델에 돌린 결과의 인덱스와 결과를 가져온다.
            idx, res = self.result_queue.get()

            # 요청된 인덱스와 결과 큐에서 나온 인덱스가 같다면 결과 반환
            if idx == self.get_idx:
                return res

            # 올바른 결과가 나올 때까지 반복
            # result_rank를 순서대로 검사 했을 때  index보다 큰  첫 번째 값의 인덱스를 반환
            insert = bisect.bisect(self.result_rank, idx)
            # result_rank[insert] = idx
            self.result_rank.insert(insert, idx)
            # result_rank[insert] = res
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
