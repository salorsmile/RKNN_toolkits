import os
import cv2
import sys
import argparse
import numpy as np

'''
    批量测试视频和图像，支持.pt .pnnx .rknn格式的模型
    用法：
    @视频
    python your_script.py --model_path your_model.rknn --video_folder /path/to/video_folder

    @图片
    python your_script.py --model_path your_model.rknn --image_folder /path/to/image_folder

'''

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:realpath.index('RKNN_toolkits') + 1]))

from py_utils.coco_utils import COCO_test_helper

# ---------------------------------------------------  配置参数 --------------------------------------------------#
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("closed", "open")


# coco_id_list = [0, 1]


# ---------------------------------------------------  批量推理 --------------------------------------------------#

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 由于门经常在画面边缘，文字标签经常不显示
# 在框下方绘制标签
def draw_bellow(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.device_id)
    elif model_path.endswith('.onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import Onnx_model_container
        model = Onnx_model_container(args.model_path)
    else:
        raise ValueError("Unsupported model format")

    return model, platform


def process_video_folder(video_folder, model, platform, co_helper):
    # 获取所有视频文件
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    # 创建输出目录（如果不存在）
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 遍历每个视频文件
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        if not os.path.exists(video_path):
            print(f"Video file {video_file} does not exist.")
            continue

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video {video_file}.")
            continue

        # 获取视频尺寸（宽和高）
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频保存对象，使用原始视频的分辨率
        output_path = os.path.join(output_dir, f'output_{os.path.splitext(video_file)[0]}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编解码器，或者根据需要改为 'XVID' 或 'H264'
        out = cv2.VideoWriter(output_path, fourcc, 30, (video_width, video_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 图像预处理（resize, padding等操作）
            img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 根据平台选择处理输入数据
            if platform in ['pytorch', 'onnx']:
                input_data = img.transpose((2, 0, 1))  # 转置为 (C, H, W)
                input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
                input_data = input_data / 255.0  # 归一化
            else:
                input_data = img

            # 模型推理
            outputs = model.run([input_data])
            boxes, classes, scores = post_process(outputs)

            # 如果有检测到框，绘制到原图
            if boxes is not None:
                draw_bellow(frame, co_helper.get_real_box(boxes), scores, classes)

            # 保存处理后的每一帧
            out.write(frame)

        # 释放资源
        cap.release()
        out.release()
        print(f"Processed video saved to {output_path}")


def process_image_folder(image_folder, model, platform, co_helper):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Image file {image_file} does not exist.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_file}.")
            continue

        img = co_helper.letter_box(im=img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.0
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)

        if boxes is not None:
            draw_bellow(img, co_helper.get_real_box(boxes), scores, classes)

        # Save the processed image
        output_img_path = os.path.join('./result', f'output_{image_file}')
        cv2.imwrite(output_img_path, img)
        print(f"Processed image saved to {output_img_path}")


def main(args):
    model, platform = setup_model(args)
    co_helper = COCO_test_helper(enable_letter_box=True)

    if args.video_folder:
        process_video_folder(args.video_folder, model, platform, co_helper)
    elif args.image_folder:
        process_image_folder(args.image_folder, model, platform, co_helper)

    model.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch process video and images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--video_folder', type=str, default=None,
                        help='Directory containing videos for batch processing')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Directory containing images for batch processing')
    parser.add_argument('--device_id', type=str, default='rk3588', help='Device ID for the RKNN model')

    args = parser.parse_args()
    main(args)
