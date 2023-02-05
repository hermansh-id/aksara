import os
import sys, shutil
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (Profile, check_file, check_img_size, check_imshow, cv2,
                           increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

path_this = os.path.abspath(os.path.dirname(__file__))
path_src = os.path.abspath(os.path.join(path_this, ".."))
path_root = os.path.abspath(os.path.join(path_src, ".."))
path_yolov5 = os.path.abspath(os.path.join(path_src, "yolov5"))
path_services = os.path.abspath(os.path.join(path_src, "services"))
sys.path.extend([path_this,path_src,path_yolov5,path_services,path_root])

class Detection:
    def __init__(self):
        imgsz=(640, 640)
        data=ROOT / 'data/coco128.yaml'
        self.weights = os.path.join(path_this,'..', 'model', 'jawa.pt')
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)

    @smart_inference_mode()
    def run(self, source):
        conf_thres=0.25
        iou_thres=0.45
        max_det=1000
        view_img=False
        save_txt=False
        save_conf=False
        save_crop=False
        classes=None
        agnostic_nms=False
        augment=False
        visualize=False
        update=False
        
        path_pred = os.path.join(path_src, 'prediction')
        if(os.path.exists(path_pred)):
            shutil.rmtree(path_pred)
        else:
            os.mkdir(path_pred)
        project = path_src
        name='prediction'
        exist_ok=True
        line_thickness=3
        hide_labels=False
        hide_conf=False
        vid_stride=1
        nosave=False
        
        
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None] 

            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = self.model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % im.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                imc = im0.copy() if save_crop else im0
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum() 
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:
                            c = int(cls)
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                im0 = annotator.result()

                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path[i] != save_path:
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
