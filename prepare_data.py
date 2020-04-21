"""A script that downloads the sintel movie and splits it into chunks and frames
(used to test out various video data loaders).
"""
import argparse
import shutil
import functools
import subprocess
from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
from typeguard import typechecked


@typechecked
@functools.lru_cache(maxsize=64, typed=False)
def get_video_duration(src_path: Path) -> float:
    assert src_path.exists(), f"Video at {src_path} not found"
    cap = cv2.VideoCapture(str(src_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_res_t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return video_res_t / video_fps


@typechecked
def re_encode_movie(
        src_movie_path: Path,
        dest_movie_path: Path,
        refresh: bool,
        keyframe_interval: int,
        fps: int,
        scale: Tuple[int, int] = None,
):
    """Re-encode a video to optimise it for usage with DALI.

    Args:
        src_movie_path: the original movie to be re-encoded
        keyframe_interval: the number of frames between keyframes
        refresh: whether to overwrite an existing video at the destination
        scale: if given, rescale the frames to have these spatial dimensions.
        fps: the output framerate of the re-encoded movie.
        dest_movie_path: the destination path where the re-encoded video will be stored

    NOTE: The basic idea behind this reencoding is to ensure that there are sufficiently
    many keyframes to allow fast random access into the video.  Following the
    recommendation from NVVL, it is considered wise to set the keyframe interval
    to reflect the expected clip size. We also encode as h.264 and use yuv420p format.
    """
    if dest_movie_path.exists() and not refresh:
        print(f"Found re-encoded movie at {dest_movie_path}, skipping...")
        return
    msg = f"Expected mp4 suffix for destination, found {dest_movie_path.suffix}"
    assert dest_movie_path.suffix == ".mp4", msg
    # Ensure that the operation atomic
    tmp_path = dest_movie_path.parent / f"{dest_movie_path.stem}.tmp.mp4"
    print(f"Re-encoding movie in h264 and saving to {tmp_path}")
    flags = (f" -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g {keyframe_interval} "
             f"-vf fps=fps={fps} -profile:v high -c:a copy ")
    cmd = f"ffmpeg -i {src_movie_path} {flags}"
    if scale:
        cmd = f"{cmd} -vf scale={scale[0]}:{scale[1]}"
    cmd = f"{cmd} {tmp_path}"
    dest_movie_path.parent.mkdir(exist_ok=True, parents=True)
    subprocess.call(cmd.split())
    # rotate temp file back only after completion to preserve atomicity
    shutil.move(tmp_path, dest_movie_path)



@typechecked
def fetch_movie(url: str, orig_movie_path: Path, refresh: bool):
    if orig_movie_path.exists() and not refresh:
        print(f"Found existing sintel movie at {url}")
        return
    print(f"Fetching original sintel movie from {url}")
    subprocess.call(["wget", url, "-O", str(orig_movie_path)])


@typechecked
def split_movie_to_chunks(
        movie_path: Path,
        movie_chunk_dir: Path,
        duration: str,
        refresh: bool,
):
    first_chunk = movie_chunk_dir / "chunk000.mp4"
    if first_chunk.exists() and not refresh:
        print(f"Found existing chunk at {first_chunk}, skipping...")
        return
    cmd = (f"ffmpeg -i {movie_path} -c copy -map 0 -segment_time {duration} "
           f"-f segment -reset_timestamps 1 {movie_chunk_dir}/chunk%03d.mp4")
    movie_chunk_dir.mkdir(exist_ok=True, parents=True)
    subprocess.call(cmd.split())


@typechecked
def split_chunks_to_frames(movie_chunk_dir: Path, frame_dir: Path, refresh: bool):
    chunks = list(movie_chunk_dir.glob("*.mp4"))
    print(f"Found {len(chunks)} movie chunks")
    for chunk in chunks:
        dest_dir = frame_dir / chunk.stem
        if dest_dir.exists() and not refresh:
            print(f"Found existing frames at {dest_dir}, skipping...")
            return
        dest_dir.mkdir(exist_ok=True, parents=True)
        cmd = f"ffmpeg -i {chunk} {dest_dir}/%05d.jpg"
        subprocess.call(cmd.split())


@typechecked
def prepare_file_list(
        movie_chunk_dir: Path,
        frame_dir: Path,
        video_list_path: Path,
        frame_list_path: Path,
        refresh: bool,
):
    if video_list_path.exists() and frame_list_path.exists() and not refresh:
        print(f"Found existing file_list at {video_list_path}, skipping...")
        return
    chunks = list(sorted(movie_chunk_dir.glob("*.mp4")))
    print(f"Found {len(chunks)} chunks, writing list to {video_list_path}")
    with open(video_list_path, "w") as f:
        for video_path in chunks:
            # use random offset times with an arbitrary label to showcase format
            label = 0
            clip_duration = 3
            video_duration = get_video_duration(video_path)
            assert video_duration > 0, "Cannot include empty videos"
            if video_duration < clip_duration:
                start_time, end_time = 0, video_duration
            else:
                end_time = np.random.rand() * (video_duration - clip_duration)
                end_time = max(clip_duration, end_time)
                start_time = end_time - clip_duration
            f.write(f"{video_path} {label} {start_time:.2f} {end_time:.2f}\n")
    frames = list(sorted(frame_dir.glob("**/*.jpg")))
    print(f"Found {len(frames)} frames, writing list to {frame_list_path}")
    with open(frame_list_path, "w") as f:
        for frame in frames:
            f.write(f"{frame}\n")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--refresh", action="store_true")
    args.add_argument("--keyframe_interval", type=int, default=16)
    args.add_argument("--task", default="prepare-videos",
                      choices=["prepare-videos", "prepare-metadata"])
    args.add_argument("--data_dir", default="data/sintel", type=Path)
    args.add_argument("--chunk_duration", default="00:00:20",
                      help=("The duration (specified as HH:MM:SS) to be used for each "
                            "timestamp."))
    args.add_argument("--src_url", default=("http://peach.themazzone.com/durian/movies/"
                                            "sintel-1024-surround.mp4"))
    args = args.parse_args()

    args.data_dir.mkdir(exist_ok=True, parents=True)
    orig_movie_path = args.data_dir / Path(args.src_url).name
    mp4_movie_path = args.data_dir / f"{orig_movie_path.stem}-reencoded.mp4"
    movie_chunk_dir = args.data_dir / f"{Path(args.src_url).stem}-chunks"
    movie_frame_dir = args.data_dir / f"{Path(args.src_url).stem}-frames"
    video_list_path = args.data_dir / "video_list.txt"
    frame_list_path = args.data_dir / "frame_list.txt"

    fetch_movie(
        url=args.src_url,
        orig_movie_path=orig_movie_path,
        refresh=args.refresh,
    )
    re_encode_movie(
        mp4_movie_path=mp4_movie_path,
        orig_movie_path=orig_movie_path,
        keyframe_interval=args.keyframe_interval,
        refresh=args.refresh,
    )
    split_movie_to_chunks(
        refresh=args.refresh,
        duration=args.chunk_duration,
        movie_path=mp4_movie_path,
        movie_chunk_dir=movie_chunk_dir,
    )
    split_chunks_to_frames(
        refresh=args.refresh,
        movie_chunk_dir=movie_chunk_dir,
        frame_dir=movie_frame_dir,
    )
    prepare_file_list(
        refresh=args.refresh,
        movie_chunk_dir=movie_chunk_dir,
        frame_dir=movie_frame_dir,
        video_list_path=video_list_path,
        frame_list_path=frame_list_path,
    )


if __name__ == "__main__":
    main()
