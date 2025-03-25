import os
import sys
import argparse
import numpy as np
import tty
import termios
import urllib
import urllib.request
import time
import numpy as np
import cv2
import math
from math import ceil
from itertools import product as product
from shapely.geometry import Polygon

from rknn.api import RKNN

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

# ---------------------------------------------------  配置参数 --------------------------------------------------#
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (800, 800)  # (width, height), such as (1280, 736)

CLASSES = ("stick")


# CLASSES = ("closed", "open")


# ---------------------------------------------------  批量推理 --------------------------------------------------#

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y


def map_points_to_original(points, img_shape=(1920, 1080), model_shape=(800, 800)):
    """
    将 800x800 推理得到的 OBB (4点坐标) 映射回 1920x1080 原图尺寸

    参数：
        points: list 或 np.ndarray，形状 (4, 2) 或 (N, 4, 2)，表示 OBB 坐标
        img_shape: tuple，原图大小 (宽, 高)
        model_shape: tuple，模型输入大小 (宽, 高)

    返回：
        np.ndarray: 映射回原图的 OBB，形状 (4, 2) 或 (N, 4, 2)
    """
    img_w, img_h = img_shape
    model_w, model_h = model_shape

    # 计算等比例缩放因子
    scale = min(model_w / img_w, model_h / img_h)

    # 计算填充大小
    pad_w = (model_w - img_w * scale) / 2  # 左右填充
    pad_h = (model_h - img_h * scale) / 2  # 上下填充

    # 确保 points 是 numpy 数组
    points = np.array(points, dtype=np.float32)

    # 处理单个 OBB（4点）的情况，转换为 (1, 4, 2)
    if points.shape == (4, 2):
        points = points[np.newaxis, :, :]  # 变成 (1, 4, 2)

    # 进行逆映射变换
    points[:, :, 0] = (points[:, :, 0] - pad_w) / scale  # x 方向
    points[:, :, 1] = (points[:, :, 1] - pad_h) / scale  # y 方向

    return points.squeeze(0)  # 如果是单个 OBB，去掉 batch 维度，返回 (4, 2)


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, angle):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.angle = angle


def rotate_rectangle(x1, y1, x2, y2, a):
    # 计算中心点坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 将角度转换为弧度
    # a = math.radians(a)
    # 对每个顶点进行旋转变换
    x1_new = int((x1 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y1_new = int((x1 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)

    x2_new = int((x2 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y2_new = int((x2 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x3_new = int((x1 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y3_new = int((x1 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x4_new = int((x2 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y4_new = int((x2 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)
    return [(x1_new, y1_new), (x3_new, y3_new), (x2_new, y2_new), (x4_new, y4_new)]


def intersection(g, p):
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        angle = sort_detectboxs[i].angle
        p1 = rotate_rectangle(xmin1, ymin1, xmax1, ymax1, angle)
        p1 = np.array(p1).reshape(-1)

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    angle2 = sort_detectboxs[j].angle
                    p2 = rotate_rectangle(xmin2, ymin2, xmax2, ymax2, angle2)
                    p2 = np.array(p2).reshape(-1)
                    iou = intersection(p1, p2)
                    if iou > NMS_THRESH:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    # 将输入向量减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def process(out, model_w, model_h, stride, angle_feature, index, scale_w=1, scale_h=1):
    class_num = len(CLASSES)
    angle_feature = angle_feature.reshape(-1)
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    out = []
    conf = conf.reshape(-1)
    for ik in range(model_h * model_w * class_num):
        if conf[ik] > OBJ_THRESH:
            w = ik % model_w
            h = (ik % (model_w * model_h)) // model_w
            c = ik // (model_w * model_h)
            xywh_ = xywh[0, :, (h * model_w) + w]  # [1,64,1]
            xywh_ = xywh_.reshape(1, 4, 16, 1)
            data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
            xywh_ = softmax(xywh_, 2)
            xywh_ = np.multiply(data, xywh_)
            xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)
            xywh_add = xywh_[:2] + xywh_[2:]
            xywh_sub = (xywh_[2:] - xywh_[:2]) / 2
            angle_feature_ = (angle_feature[index + (h * model_w) + w] - 0.25) * 3.1415927410125732
            angle_feature_cos = math.cos(angle_feature_)
            angle_feature_sin = math.sin(angle_feature_)
            xy_mul1 = xywh_sub[0] * angle_feature_cos
            xy_mul2 = xywh_sub[1] * angle_feature_sin
            xy_mul3 = xywh_sub[0] * angle_feature_sin
            xy_mul4 = xywh_sub[1] * angle_feature_cos
            xy = xy_mul1 - xy_mul2, xy_mul3 + xy_mul4
            xywh_1 = np.array([(xy_mul1 - xy_mul2) + w + 0.5, (xy_mul3 + xy_mul4) + h + 0.5, xywh_add[0], xywh_add[1]])
            xywh_ = xywh_1 * stride
            xmin = (xywh_[0] - xywh_[2] / 2) * scale_w
            ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
            xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
            ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
            box = DetectBox(c, conf[ik], xmin, ymin, xmax, ymax, angle_feature_)
            out.append(box)
    return out


def draw_score_threshold(image, text_color=(0, 255, 0), font_scale=0.8, thickness=2):
    """
    在图像左上角绘制得分阈值。

    参数:
        image: 输入的图像 (numpy数组).
        text_color: 文本颜色 (BGR格式, 默认绿色).
        font_scale: 字体大小 (默认0.8).
        thickness: 文本厚度 (默认2).
    """
    # 构造文本内容
    text = f"OBJ_THRESH: {OBJ_THRESH:.2f}"

    # 获取文本大小
    (label_width, label_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # 设置文本位置 (左上角，留一些边距)
    text_x = 10  # 水平偏移
    text_y = label_height + 10  # 垂直偏移

    # 绘制文本
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


def wait_for_space():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            ch = sys.stdin.read(1)
            if ch == ' ':
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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
        model = RKNN_model_container(args)
        if args.print_sdk_version:
            model.print_sdk_version()
        if args.print_model_perf:
            model.print_model_perf()
        print("\n\n\n\033[31m 按空格键继续..\033[0m\n\n\n")
        wait_for_space()
        print("\n\n\n\033[32m 模型推理..\033[0m\n\n\n")
    elif model_path.endswith('.onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import Onnx_model_container
        model = Onnx_model_container(args.model_path)
    else:
        raise ValueError("Unsupported model format")

    return model, platform


# def process_video_folder(video_folder, model, platform, tag):
#     # 获取所有视频文件
#     video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
#
#     # 创建输出目录（如果不存在）
#     output_dir = os.path.join('./result/video', tag)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#     # 遍历每个视频文件
#     for video_file in video_files:
#         video_path = os.path.join(video_folder, video_file)
#         if not os.path.exists(video_path):
#             print(f"Video file {video_file} does not exist.")
#             continue
#
#         # 打开视频文件
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"Failed to open video {video_file}.")
#             continue
#
#         # 获取视频尺寸（宽和高）
#         video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#         # 创建视频保存对象，使用原始视频的分辨率
#         output_path = os.path.join(output_dir, f'output_{os.path.splitext(video_file)[0]}.mp4')
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编解码器，或者根据需要改为 'XVID' 或 'H264'
#         out = cv2.VideoWriter(output_path, fourcc, 30, (video_width, video_height))
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # 图像预处理（resize, padding等操作）
#             img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#             # 根据平台选择处理输入数据
#             if platform in ['pytorch', 'onnx']:
#                 input_data = img.transpose((2, 0, 1))  # 转置为 (C, H, W)
#                 input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
#                 input_data = input_data / 255.0  # 归一化
#             else:
#                 input_data = img
#
#             # 模型推理
#             outputs = model.run([input_data])
#             boxes, classes, scores = post_process(outputs)
#
#             # 如果有检测到框，绘制到原图
#             if boxes is not None:
#                 draw_bellow(frame, co_helper.get_real_box(boxes), scores, classes)
#
#             # 保存处理后的每一帧
#             out.write(frame)
#
#         # 释放资源
#         cap.release()
#         out.release()
#         print("\n\033[32mProcessed video saved to {0}\033[0m\n".format(output_path))


def process_image_folder(image_folder, model, platform, tag):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

    # print(image_files)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        # print(image_path)

        # 创建输出目录（如果不存在）
        output_dir = os.path.join('./result/image', tag)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(image_path):
            print(f"Image file {image_file} does not exist.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_file}.")
            continue

        file_name = os.path.basename(image_path)
        file_root, ext = os.path.splitext(file_name)
        new_file_name = file_name  # 默认保持原文件名

        # 获取原始图像尺寸
        original_height, original_width = img.shape[:2]

        # 进行缩放并去除黑色填充，推理时使用 640x640
        # img_resized = co_helper.letter_box(im=img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        # img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img.copy(), (IMG_SIZE[1], IMG_SIZE[0]),
                                                                           114)  # letterbox缩放
        infer_img = letterbox_img[..., ::-1]  # BGR2RGB

        if platform in ['pytorch', 'onnx']:
            input_data = infer_img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.0
        else:
            input_data = infer_img

        # 模型推理
        outputs = model.run([input_data])
        # 后处理

        print(f"outputs.shape -> {outputs.size()}")

        results = []
        for x in outputs[:-1]:

            print(f"x.shape -> {x.shape}")
            print(f"x.shape[2] 20 40 80 ? -> {x.shape}")
            index, stride = 0, 0
            if x.shape[2] == 20:
                stride = 32
                index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
            if x.shape[2] == 40:
                stride = 16
                index = 20 * 4 * 20 * 4
            if x.shape[2] == 80:
                stride = 8
                index = 0
            feature = x.reshape(1, 79, -1)

            print(f"feature.shape -> {feature.shape}..")

            results = process(feature, x.shape[3], x.shape[2], stride, outputs[-1], index)
            results = results + results
        predbox = NMS(results)

        formatted_points = ''

        for index in range(len(predbox)):
            xmin = int((predbox[index].xmin - offset_x) / aspect_ratio)
            ymin = int((predbox[index].ymin - offset_y) / aspect_ratio)
            xmax = int((predbox[index].xmax - offset_x) / aspect_ratio)
            ymax = int((predbox[index].ymax - offset_y) / aspect_ratio)
            classId = predbox[index].classId
            score = predbox[index].score
            angle = predbox[index].angle

            if score > OBJ_THRESH:
                points = rotate_rectangle(xmin, ymin, xmax, ymax, angle)

                print(f"points before map -> {points}")
                # o -> WH   i -> WH
                points = map_points_to_original(points, (original_width, original_height), IMG_SIZE)

                print(f"points after map -> {points}")

                cv2.polylines(img, [np.asarray(points, dtype=int)], True, (0, 255, 0), 1)

                ptext = (xmin, ymin)
                title = CLASSES[classId] + "%.2f" % score
                cv2.putText(img, title, ptext, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

                formatted_points = "_".join(f"({x}, {y})" for x, y in [points[0], points[2], points[1], points[3]])

        new_file_name = f"{file_root}_" + formatted_points + ext

        # Convert the image back to BGR before saving
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Save the processed image with original size
        output_img_path = os.path.join(output_dir, f'output_{new_file_name}')
        cv2.imwrite(output_img_path, img)
        print(f"Processed image saved to {output_img_path}")


def main(args):
    model, platform = setup_model(args)

    if args.video_folder:
        pass
        # process_video_folder(args.video_folder, model, platform, args.tag)
    elif args.image_folder:
        print(args.image_folder)
        process_image_folder(args.image_folder, model, platform, args.tag)

    model.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch process video and images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--tag', type=str, default='stick-800', help='result tag')
    parser.add_argument('--video_folder', type=str, default=None,
                        help='Directory containing videos for batch processing')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Directory containing images for batch processing')
    parser.add_argument('--target', type=str, default='rk3588', help='Target plantom for the RKNN model')
    parser.add_argument('--device_id', type=str, default='rk3588',
                        help='Device ID for the RKNN model')  # 如果ADB链接多个设备 通过list查看设备号
    # 进行性能评估时是否开启debug模式。在debug模式下，可以获取到每一层的运行时间，否则只能获取模型运行的总时间。默认值为False。
    parser.add_argument('--perf_debug', type=bool, default=True, help='Display each layer s runtime cost')  #
    # 是否进入内存评估模式。进入内存评估模式后，可以调用eval_memory接口获取模型运行时的内存使用情况。默认值为False。
    parser.add_argument('--eval_mem', type=bool, default=True, help='Print model memory allocation')

    parser.add_argument('--print_model_perf', type=bool, default=True, help='打印性能评估信息')
    parser.add_argument('--print_sdk_version', type=bool, default=False, help='打印SDK API和驱动版本号')
    args = parser.parse_args()
    main(args)
