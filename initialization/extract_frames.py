# BioPhysGS release: require explicit input/output locations.
import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fps=1):
    """
    从视频中每秒抽取指定帧数并保存为PNG格式

    Args:
        video_path: 视频文件路径
        output_dir: 输出图片目录
        fps: 每秒抽取的帧数，默认为1
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"视频FPS: {video_fps:.2f}")
    print(f"总帧数: {total_frames}")
    print(f"视频时长: {duration:.2f}秒")
    print(f"抽帧频率: 每秒{fps}帧")

    # 计算帧间隔
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔保存帧
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{saved_count:06d}.png")
            cv2.imwrite(output_path, frame)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"已保存 {saved_count} 帧...")

        frame_count += 1

    cap.release()
    print(f"完成! 总共保存了 {saved_count} 帧到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中抽取帧")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--fps", type=float, default=3, help="每秒抽取的帧数")

    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps)
