"""A script that downloads the sintel movie and splits it into chunks and frames
(used to test out various video data loaders.)
"""
import argparse
import subprocess
from pathlib import Path


def fetch_movie(url, orig_movie_path, refresh):
    if orig_movie_path.exists() and not refresh:
        print(f"Found existing sintel movie at {url}")
        return
    print(f"Fetching original sintel movie from {url}")
    subprocess.call(["wget", url, "-O", str(orig_movie_path)])


def re_encode_movie(orig_movie_path, mp4_movie_path, refresh):
    if mp4_movie_path.exists() and not refresh:
        print(f"Found re-encoded mp4 sintel movie at {mp4_movie_path}, skipping...")
        return
    print(f"Re-encoding sintel movie in h264/mp4 and saving to {mp4_movie_path}")
    cmd = f"ffmpeg -i {orig_movie_path} -c:v libx264 -c:a copy {mp4_movie_path}"
    subprocess.call(cmd.split())


def split_movie_to_chunks(orig_movie_path, duration, refresh, movie_chunk_dir):
    first_chunk = movie_chunk_dir / "chunk000.mp4"
    if first_chunk.exists() and not refresh:
        print(f"Found existing chunk at {first_chunk}, skipping...")
        return
    cmd = (f"ffmpeg -i {orig_movie_path} -c copy -map 0 -segment_time {duration} "
           f"-f segment -reset_timestamps 1 {movie_chunk_dir}/chunk%03d.mp4")
    movie_chunk_dir.mkdir(exist_ok=True, parents=True)
    subprocess.call(cmd.split())


def split_chunks_to_frames(movie_chunk_dir, frame_dir, refresh):
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


def prepare_file_list(movie_chunk_dir, frame_dir, video_list_path, frame_list_path,
                      refresh):
    if video_list_path.exists() and frame_list_path.exists() and not refresh:
        print(f"Found existing file_list at {video_list_path}, skipping...")
        return
    chunks = list(sorted(movie_chunk_dir.glob("*.mp4")))
    with open(video_list_path, "w") as f:
        for video_path in chunks:
            # use arbitrary start/end frames for now
            label, start_frame, end_frame = 1, "A", "A"
            f.write(f"{video_path} {start_frame} {end_frame} {label}\n")
    frames = list(sorted(frame_dir.glob("**/*.jpg")))
    print(f"Found {len(frames)} frames, writing list to {frame_list_path}")
    with open(frame_list_path, "w") as f:
        for frame in frames:
            f.write(f"{frame}\n")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--refresh", action="store_true")
    args.add_argument("--task", default="prepare-videos",
                      choices=["prepare-videos", "prepare-metadata"])
    args.add_argument("--data_dir", default="data/sintel")
    args.add_argument("--chunk_duration", default="00:00:20", help="chunk duration")
    args.add_argument("--src_url", default=("http://peach.themazzone.com/durian/movies/"
                                            "sintel-1024-surround.mp4"))
    args = args.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    orig_movie_path = data_dir / Path(args.src_url).name
    movie_chunk_dir = data_dir / f"{Path(args.src_url).stem}-chunks"
    movie_frame_dir = data_dir / f"{Path(args.src_url).stem}-frames"
    video_list_path = data_dir / "video_list.txt"
    frame_list_path = data_dir / "frame_list.txt"
    fetch_movie(
        url=args.src_url,
        orig_movie_path=orig_movie_path,
        refresh=args.refresh,
    )
    split_movie_to_chunks(
        refresh=args.refresh,
        duration=args.chunk_duration,
        orig_movie_path=orig_movie_path,
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
