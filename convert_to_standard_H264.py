import os
import cv2


def convert_to_h264_mp4(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in video_files:
        input_video_path = os.path.join(input_folder, video_file)
        output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_h264.mp4")

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create VideoWriter object (H.264 codec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v instead of H264

        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        print(f"Processing video: {video_file}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        print(f"Converted {video_file} to {output_video_path}")


if __name__ == "__main__":
    input_folder = r'D:\_proj\Python\RKNN_toolkits\data\video'  # 输入视频文件夹
    output_folder = r'D:\_proj\Python\RKNN_toolkits\data\video-t'  # 输出视频文件夹
    convert_to_h264_mp4(input_folder, output_folder)
