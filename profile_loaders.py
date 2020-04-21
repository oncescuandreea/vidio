"""A naive benchmark on the efficiency of different data loaders for video:

* A simple image loader operating on JPEGs
* DALI (https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)
* VideoClips
    (https://github.com/pytorch/vision/blob/master/torchvision/datasets/video_utils.py)

ipy compare_loaders.py -- --loader_type dali
"""

import os
import time
import pickle
import socket
import hashlib
import logging
import argparse
import subprocess
from typing import Dict, List, Iterable
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import ffmpeg
import matplotlib
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from typeguard import typechecked
from torchvision.models.video import r2plus1d_18
from torchvision.datasets.video_utils import VideoClips

from dali_video_loader import DaliVideoLoader


class FrameDataset:

    def __init__(self, frame_list_path, clip_length_in_frames, stride):
        self.stride = stride
        self.clip_length_in_frames = clip_length_in_frames
        with open(frame_list_path, "r") as f:
            self.frame_paths = f.read().splitlines()

    def __getitem__(self, idx):
        start = idx * self.clip_length_in_frames
        stop = (idx + 1) * self.clip_length_in_frames
        im_paths = self.frame_paths[start:stop]
        ims = []
        for im_path in im_paths:
            im = np.array(Image.open(im_path))
            ims.append(im)
        return {"clips": self.video_to_tensor(np.array(ims))}

    def __len__(self):
        return len(self.frame_paths) // self.clip_length_in_frames

    def video_to_tensor(self, vid):
        """Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape
        (C x T x H x W)

        Args:
            vid (numpy.ndarray): Video to be converted to tensor.
        Returns:
            Tensor: Converted video.
        """
        return torch.from_numpy(vid.transpose([3, 0, 1, 2]))


class ClipDataset:
    def __init__(self, video_paths, clip_length_in_frames, stride,
                 frame_rate, refresh, cache_dir):

        self.frame_rate = frame_rate
        self.clip_length_in_frames = clip_length_in_frames
        self.stride = stride
        self.video_paths = video_paths
        fname = f"fps-{frame_rate}-clip_length-{clip_length_in_frames}-stride{stride}"
        video_str_bytes = '-'.join(sorted(video_paths)).encode("utf-8")
        hashed = hashlib.sha256(video_str_bytes).hexdigest()
        fname += f"num-videos{len(video_paths)}-{hashed}"
        cached_clips_path = Path(cache_dir) / fname
        if cached_clips_path.exists() and not refresh:
            print(f"Reloading cached clips object")
            with open(cached_clips_path, "rb") as f:
                self.video_clips = pickle.load(f)
        else:
            print(f"Building new video clips object")
            self.video_clips = VideoClips(
                frame_rate=frame_rate,
                video_paths=video_paths,
                frames_between_clips=stride,
                clip_length_in_frames=clip_length_in_frames,
            )
            cached_clips_path.parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing object to cache at {cached_clips_path}")
            with open(cached_clips_path, "wb") as f:
                pickle.dump(self.video_clips, f)

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        return video

    def __len__(self):
        return self.video_clips.num_clips()


class Profiler:
    @typechecked
    def __init__(
            self,
            vis: bool,
            disp_fig: bool,
            include_model: bool,
            show_gpu_utilization: bool,
            max_clips: int,
            imsz: int,
            loader_type: str,
            loader: Iterable,
            logger: logging.Logger,
    ):
        self.vis = vis
        self.imsz = imsz
        self.loader = loader
        self.disp_fig = disp_fig
        self.loader_type = loader_type
        self.include_model = include_model
        self.show_gpu_utilization = show_gpu_utilization
        self.max_clips = max_clips
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        if self.include_model:
            self.model = r2plus1d_18(pretrained=False, progress=True)
            self.model = self.model.to(self.device)
        self.logger = logger
        self.logger.info(f"{loader_type} profiler, include_model: {self.include_model}")

    @typechecked
    def vis_sequence(self, sequence: torch.Tensor):
        columns = 4
        rows = (sequence.shape[0] + 1) // (columns)
        figsize = (32, (16 // columns) * rows)
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(rows, columns)
        for j in range(rows * columns):
            plt.subplot(gs[j])
            plt.axis("off")
            im = sequence[:, j].permute(1, 2, 0).cpu().numpy()
            im = im - im.min()
            im = im / im.max()
            plt.imshow(im)
        if self.disp_fig:
            from zsvision.zs_iterm import zs_dispFig
            zs_dispFig()

    def run(self):
        tic = time.time()
        hz, warmup_batches, eps = 0, 5, 1E-7
        for ii, minibatch in enumerate(self.loader):
            if ii > self.max_clips:
                return
            if self.loader_type == "dali":
                assert len(minibatch) == 1, "expected single device pipeline"
                clips = minibatch[0]["data"]
                clips = clips.permute(0, 4, 1, 2, 3)
            elif self.loader_type == "videoclip":
                clips = minibatch
                clips = clips.permute(0, 4, 1, 2, 3)
            elif self.loader_type == "frames":
                clips = minibatch["clips"]
            clips = clips.float()
            if self.vis:
                self.vis_sequence(clips[0].cpu())
            if self.show_gpu_utilization and ii % 10 == 0:
                os.system("nvidia-smi")
            # This should be a no-op for DALI (on-device)
            clips = clips.to(self.device)
            size = [clips.shape[2], self.imsz, self.imsz]
            clips = torch.nn.functional.interpolate(clips, size=size)
            msg = f"[{self.loader_type}] {ii}/{len(self.loader)} [{clips.shape}]"
            delta = max(time.time() - tic, eps)
            hz = 1 / delta
            clip_hz = clips.shape[0] / delta
            msg += f" batch hz: {hz:.2f}, clip hz: {clip_hz:.2f}, "
            if ii == warmup_batches:
                avg_timer = time.time()
                seen_clips = 0
                rolling_avg_clip_hz = []
            elif ii > warmup_batches:
                delta = max(time.time() - avg_timer, eps)
                seen_clips += clips.shape[0]
                avg_hz = (ii - warmup_batches) / delta
                avg_clip_hz = seen_clips / delta
                if len(rolling_avg_clip_hz) == 3:
                    rolling_avg_clip_hz.pop(0)
                rolling_avg_clip_hz.append(avg_clip_hz)
                msg += f"batch hz: (avg) {avg_hz:.2f}, "
                msg += f"clip hz: (avg) {avg_clip_hz:.2f}, "
                msg += f"(std) {np.std(rolling_avg_clip_hz):.2f}"
            self.logger.info(msg)
            tic = time.time()
            if self.include_model:
                with torch.no_grad():
                    self.model(clips)


@typechecked
def get_gpu_info_path(log_dir: Path):
    hostname = socket.gethostname()
    return log_dir / f"gpu-info-for-{hostname}.txt"


@typechecked
def get_GPU_info(log_dir: Path, refresh_meta: bool):
    dest_path = get_gpu_info_path(log_dir=log_dir)
    if dest_path.exists() and not refresh_meta:
        return
    cmd = f"nvidia-smi -f {dest_path}"
    subprocess.call(cmd.split())


@typechecked
def get_video_info(log_dir: Path, video_list: Path, refresh_meta: bool):
    with open(video_list, "r") as f:
        rows = f.read().splitlines()
    video_paths = [x.split()[0] for x in rows]
    # create a hash from the video paths to ensure the cache is not stale
    hashable_bytes = "".join(sorted(video_paths)).encode("utf-8")
    hash_str = hashlib.sha256(hashable_bytes).hexdigest()[:8]
    dest_path = log_dir / f"video-meta-{hash_str}.log"
    if dest_path.exists() and not refresh_meta:
        return

    dims = {"height": [], "width": []}
    for video_path in video_paths:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if
                            stream['codec_type'] == 'video'), None)
        for key in dims:
            dims[key].append(video_stream[key])
    for key, vals in dims.items():
        msg = f"Expected videos to have the same {key}, but found {set(vals)}"
        assert len(set(vals)) == 1, msg
    height, width = dims["height"][0], dims["width"][0]

    with open(dest_path, "w") as f:
        f.write(f"height: {height}, width: {width}\n")


def build_loaders(
        cache_dir: Path,
        frame_list: Path,
        video_list: Path,
        shuffle: bool,
        refresh: bool,
        stride: int,
        seed: int,
        batch_size: int,
        num_workers: int,
        frame_rate: int,
        clip_length_in_frames: int,
        loader_types: List[str],
) -> Dict:
    loaders = {}
    for loader_type in loader_types:
        if loader_type == "frames":
            dataset = FrameDataset(
                stride=stride,
                frame_list_path=frame_list,
                clip_length_in_frames=clip_length_in_frames,
            )
            loader = DataLoader(
                drop_last=True,
                dataset=dataset,
                pin_memory=True,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        elif loader_type == "videoclip":
            with open(video_list, "r") as f:
                rows = f.read().splitlines()
                video_paths = [x.split()[0] for x in rows]

            dataset = ClipDataset(
                stride=stride,
                video_paths=video_paths,
                frame_rate=frame_rate,
                clip_length_in_frames=clip_length_in_frames,
                cache_dir=cache_dir,
                refresh=refresh,
            )
            loader = DataLoader(
                shuffle=shuffle,
                drop_last=True,
                dataset=dataset,
                pin_memory=True,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        elif loader_type == "dali":
            # use the first visible gpu for loading
            device_id = 0
            loader = DaliVideoLoader(
                device_id=device_id,
                video_list=video_list,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                clip_length_in_frames=clip_length_in_frames,
                initial_prefetch_size=1,
                seed=seed,
            )
        print(f"finished creating {loader_type} dataloader")
        loaders[loader_type] = loader
    return loaders


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--imsz", type=int, default=224)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--frame_rate", type=int, default=25)
    args.add_argument("--shuffle", type=int, default=1)
    args.add_argument("--max_clips", type=int, default=25)
    args.add_argument("--loader_types", nargs='+', default="dali")
    args.add_argument("--stride", type=int, default=1)
    args.add_argument("--refresh", action="store_true")
    args.add_argument("--show_gpu_utilization", action="store_true")
    args.add_argument("--sintel", action="store_true")
    args.add_argument("--refresh_meta", action="store_true")
    args.add_argument("--include_model", action="store_true",
                      help="run a GPU-based model as part of the profiling ")
    args.add_argument("--vis", action="store_true")
    args.add_argument("--disp_fig", action="store_true")
    args.add_argument("--clip_length_in_frames", type=int, default=16)
    args.add_argument("--cache_dir", type=Path, default="data/video_clips_caches")
    args.add_argument("--log_dir", default="data/logs", type=Path)
    args.add_argument("--video_list", type=Path, default="data/sintel/video_list.txt")
    args.add_argument("--frame_list", type=Path, default="data/sintel/frame_list.txt")
    args = args.parse_args()

    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    log_path = args.log_dir / f"{timestamp}.txt"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger = logging.getLogger("profiler")
    logger.addHandler(console)

    # prevent ipython from duplicating logs
    if len(logger.handlers) > 1:
        logger.handlers = logger.handlers[:1]

    logger.info(f"Profiling {args.loader_types}")
    logger.info(f"Number of CPU workers - {args.num_workers}")
    logger.info(f"Frame rate - {args.frame_rate} fps")
    get_GPU_info(log_dir=args.log_dir, refresh_meta=args.refresh_meta)
    get_video_info(
        log_dir=args.log_dir,
        video_list=args.video_list,
        refresh_meta=args.refresh_meta,
    )

    loaders = build_loaders(
        refresh=args.refresh,
        seed=args.seed,
        shuffle=bool(args.shuffle),
        batch_size=args.batch_size,
        stride=args.stride,
        video_list=args.video_list,
        frame_list=args.frame_list,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        frame_rate=args.frame_rate,
        loader_types=args.loader_types,
        clip_length_in_frames=args.clip_length_in_frames,
    )
    for loader_type, loader in loaders.items():
        profiler = Profiler(
            vis=args.vis,
            loader=loader,
            loader_type=loader_type,
            disp_fig=args.disp_fig,
            max_clips=args.max_clips,
            include_model=args.include_model,
            show_gpu_utilization=args.show_gpu_utilization,
            logger=logger,
            imsz=args.imsz,
        )
        profiler.run()


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
