import sys

sys.path.insert(0, "./yolov5")

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # set True to speed up constant image size inference

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = "{}{:d}".format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )
    return img


def save_bounding_boxes_to_text(frame_idx, outputs, txt_path):
    for j, output in enumerate(outputs):
        bbox_left = output[0]
        bbox_top = output[1]
        bbox_w = output[2]
        bbox_h = output[3]
        identity = output[-1]
        with open(txt_path, "a") as f:
            f.write(
                ("%g " * 10 + "\n")
                % (
                    frame_idx,
                    identity,
                    bbox_left,
                    bbox_top,
                    bbox_w,
                    bbox_h,
                    -1,
                    -1,
                    -1,
                    -1,
                )
            )  # label format


def convert_detections_to_updated_tracks(det, img, im0, names, deepsort):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    bbox_xywh = []
    confs = []

    # Adapt detections to deep sort input format
    for *xyxy, conf, cls in det:
        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
        obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(obj)
        confs.append([conf.item()])

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    # Pass detections to deepsort
    return deepsort.update(xywhs, confss, im0)


def detect(opt):
    out, source, weights, view_img, save_txt, save_img, imgsz = (
        opt.output,
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.save_img,
        opt.img_size,
    )
    webcam = (
        source == "0"
        or source.startswith("rtsp")
        or source.startswith("http")
        or source.endswith(".txt")
    )

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE // 4,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )

    # Initialize
    device = torch.device("cuda:" + opt.device)
    os.makedirs(out, exist_ok=True)  # make new output folder
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)["model"].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    view_img = True
    dataset = LoadImages(
        source, img_size=imgsz, rank=opt.rank, num_ranks=opt.num_ranks, output_dir=out
    )

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != "cpu" else None

    save_path = str(Path(out))
    batch_imgs, batch_meta_data = [], []
    batch_size = 32
    last_frame_idx = 0
    current_path = ""
    for new_frame_idx, (new_path, new_img, new_im0s, new_vid_cap) in enumerate(dataset):
        if len(batch_imgs) == batch_size or (
            len(batch_imgs) > 0 and new_path != current_path
        ):  # new video
            batch_imgs = torch.stack(batch_imgs)
            # Inference
            t1 = time_synchronized()

            for batch_index, pred in enumerate(
                model(batch_imgs, augment=opt.augment)[0]
            ):
                img = batch_imgs[batch_index].unsqueeze(0)
                frame_idx, path, im0s, vid_cap = batch_meta_data[batch_index]
                video_name = path.replace(".mp4", "").split("/")[-1]
                txt_path = str(Path(out)) + f"/results_{video_name}.txt"
                # Apply NMS
                pred = non_max_suppression(
                    pred.unsqueeze(0),
                    opt.conf_thres,
                    opt.iou_thres,
                    classes=opt.classes,
                    agnostic=opt.agnostic_nms,
                )
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    im0 = im0s
                    save_path = str(Path(out) / Path(path).name)

                    if det is not None and len(det):
                        outputs = convert_detections_to_updated_tracks(
                            det, img, im0, names, deepsort
                        )

                        # draw boxes for visualization
                        if save_img and len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]
                            draw_boxes(im0, bbox_xyxy, identities)

                        # Write MOT compliant results to file
                        if save_txt and len(outputs) != 0:
                            save_bounding_boxes_to_text(frame_idx, outputs, txt_path)

                    else:
                        deepsort.increment_ages()

                    # Save results (image with detections)
                    if not save_img:
                        continue
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*opt.fourcc),
                            fps,
                            (w, h),
                        )
                    vid_writer.write(im0)
            batch_imgs, batch_meta_data = [], []

        if new_path in videos_already_done:
            continue
        img = torch.from_numpy(new_img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        batch_imgs.append(img)
        batch_meta_data.append(
            [new_frame_idx - last_frame_idx, new_path, new_im0s, new_vid_cap]
        )
        if current_path != new_path:
            last_frame_idx = new_frame_idx
            deepsort.reset()
        current_path = new_path

    if save_txt or save_img:
        print("Results saved to %s" % os.getcwd() + os.sep + out)
        if platform == "darwin":  # MacOS
            os.system("open " + save_path)

    print("Done. (%.3fs)" % (time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov5/weights/yolov5s.pt", help="model.pt path"
    )
    # file/folder, 0 for webcam
    parser.add_argument("--source", type=str, default="inference/images", help="source")
    parser.add_argument(
        "--output", type=str, default="inference/output", help="output folder"
    )  # output folder
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.2, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec (verify ffmpeg support)",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-img", action="store_true", help="save results to *.mp4")
    # class 0 is person
    parser.add_argument(
        "--classes", nargs="+", type=int, default=[0], help="filter by class"
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--rank",
        "-r",
        type=int,
        default=0,
        help="The rank of this process in videos dealing out",
    )
    parser.add_argument(
        "--num-ranks", "-nr", type=int, default=0, help="The total number of processes"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument(
        "--config_deepsort",
        type=str,
        default="deep_sort_pytorch/configs/deep_sort.yaml",
    )
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
