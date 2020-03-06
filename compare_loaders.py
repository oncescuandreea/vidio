"""A naive benchmark on the efficiency of different data loaders for video:

* A simple image loader operating on JPEGs
* VideoClips (https://github.com/pytorch/vision/blob/master/torchvision/datasets/video_utils.py)
* DALI (https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)

%run -i compare_loaders.py --loader_type dali --disp_fig
"""

import argparse
import glob
import hashlib
import logging
import os
import pickle
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from matplotlib import pyplot as plt
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.video_utils import VideoClips
from torchvision.models.video import r2plus1d_18

matplotlib.use("Agg")




try:
    from zsvision.zs_iterm import zs_dispFig
except ImportError:
    msg = ("If you are using iterm2 and want to visualise the results in the terminal"
           "then `pip instal zsvision`.  Otherwise, ignore this.")


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


class DaliVideoPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, file_list, shuffle,
                 initial_prefetch_size, clip_length_in_frames):
        super().__init__(batch_size, num_threads, device_id, seed=0)
        self.input = ops.VideoReader(
            device="gpu",
            shard_id=0,
            num_shards=1,
            file_list=file_list,
            enable_frame_num=True,
            enable_timestamps=False,
            random_shuffle=shuffle,
            initial_fill=initial_prefetch_size,
            sequence_length=clip_length_in_frames,
        )

    def define_graph(self):
        output = self.input(name="Reader")
        return output


class Profiler:
    def __init__(self, loader_type, loader, vis, disp_fig, include_model, max_clips,
                 imsz, logger, show_gpu_utilization):
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

    def vis_sequence(self, sequence):
        columns = 4
        rows = (sequence.shape[0] + 1) // (columns)
        figsize = (32, (16 // columns) * rows)
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(rows, columns)
        for j in range(rows * columns):
            plt.subplot(gs[j])
            plt.axis("off")
            plt.imshow(sequence[j])
        if self.disp_fig:
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

def get_GPU_info(args):
    bashCommand = f"nvidia-smi -f {str(args.log_dir)}/info.txt"
    subprocess.call(bashCommand.split())

def get_video_info(args):
    with open(args.video_list, "r") as f:
        rows = f.read().splitlines()
        video_paths = [x.split()[0] for x in rows]

    current_path = os.getcwd()
    log_path = Path(args.log_dir)
    os.chdir(log_path)
    list_logs = glob.glob(os.getcwd()+'/ffprobe*.log')
    for path in list_logs:
        os.remove(path)
    os.chdir(current_path)
    bashCommand = f"ffprobe {video_paths[0]} -report"
    subprocess.call(bashCommand.split())
    list_logs = glob.glob(current_path+'/ffprobe*.log')
    shutil.move(list_logs[0], str(log_path)+"/"+list_logs[0].split("/")[-1])


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--imsz", type=int, default=224)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--frame_rate", type=int, default=24)
    args.add_argument("--shuffle", type=int, default=1)
    args.add_argument("--max_clips", type=int, default=25)
    args.add_argument("--loader_types", nargs='*', default="dali")
    args.add_argument("--stride", type=int, default=1)
    args.add_argument("--refresh", action="store_true")
    args.add_argument("--show_gpu_utilization", action="store_true")
    args.add_argument("--sintel", action="store_true")
    args.add_argument("--include_model", action="store_true",
                      help="run a GPU-based model as part of the profiling ")
    args.add_argument("--vis", action="store_true")
    args.add_argument("--disp_fig", action="store_true")
    args.add_argument("--clip_length_in_frames", type=int, default=16)
    args.add_argument("--cache_dir", default="data/video_clips_caches")
    args.add_argument("--log_dir", default="data/logs")
    args.add_argument("--video_list", default="data/sintel/video_list.txt")
    args.add_argument("--frame_list", default="data/sintel/frame_list.txt")
    args = args.parse_args()

    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    log_path = Path(args.log_dir) / f"{timestamp}.txt"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger = logging.getLogger("profiler")
    logger.addHandler(console)
    logger.info(f"Profiling {args.loader_types}")
    logger.info(f"Number of CPU workers - {args.num_workers}")
    logger.info(f"Frame rate - {args.frame_rate} fps")
    get_GPU_info(args)
    get_video_info(args)
    for loader_type in args.loader_types:
        if loader_type == "frames":
            dataset = FrameDataset(
                stride=args.stride,
                frame_list_path=args.frame_list,
                clip_length_in_frames=args.clip_length_in_frames,
            )
            loader = DataLoader(
                drop_last=True,
                dataset=dataset,
                pin_memory=True,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

        elif loader_type == "videoclip":
            with open(args.video_list, "r") as f:
                rows = f.read().splitlines()
                video_paths = [x.split()[0] for x in rows]

            dataset = ClipDataset(
                stride=args.stride,
                video_paths=video_paths,
                frame_rate=args.frame_rate,
                clip_length_in_frames=args.clip_length_in_frames,
                cache_dir=args.cache_dir,
                refresh=args.refresh,
            )
            loader = DataLoader(
                shuffle=args.shuffle,
                drop_last=True,
                dataset=dataset,
                pin_memory=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        elif loader_type == "dali":
            pipe = DaliVideoPipeline(
                device_id=0,
                shuffle=args.shuffle,
                file_list=args.video_list,
                batch_size=args.batch_size,
                initial_prefetch_size=1,
                num_threads=args.num_workers,
                clip_length_in_frames=args.clip_length_in_frames,
            )
            pipe.build()
            loader = DALIGenericIterator(
                pipelines=[pipe],
                output_map=["data", "label", "frame_num"],
                size=pipe.epoch_size("Reader"),
            )
            DALIGenericIterator.__len__ = lambda x: pipe.epoch_size("Reader")
        print(f"finished creating {loader_type} dataloader")

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
    main()
