"""A lighteight wrapper around the NVIDIA Dali video pipeline for use with PyTorch.

Notes:
To use DALI, the videos must be:
* H.264 encoded
"""

from pathlib import Path

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typeguard import typechecked


class DaliVideoPipeline(Pipeline):
    @typechecked
    def __init__(
            self,
            batch_size: int,
            num_threads: int,
            device_id: int,
            file_list: Path,
            shuffle: bool,
            initial_prefetch_size: int,
            clip_length_in_frames: int,
            seed: int,
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )
        self.input = ops.VideoReader(
            device="gpu",
            shard_id=0,
            num_shards=1,
            file_list=file_list,
            enable_frame_num=True,
            enable_timestamps=True,
            random_shuffle=shuffle,
            initial_fill=initial_prefetch_size,
            sequence_length=clip_length_in_frames,
        )

    def define_graph(self):
        output = self.input(name="Reader")
        return output


class DaliVideoLoader:
    @typechecked
    def __init__(
            self,
            video_list: Path,
            shuffle: bool,
            device_id: int,
            batch_size: int,
            num_workers: int,
            clip_length_in_frames: int,
            initial_prefetch_size: int,
            seed: int,
    ):
        self.pipe = DaliVideoPipeline(
            device_id=device_id,
            shuffle=shuffle,
            file_list=video_list,
            batch_size=batch_size,
            num_threads=num_workers,
            initial_prefetch_size=initial_prefetch_size,
            clip_length_in_frames=clip_length_in_frames,
            seed=seed,
        )
        self.pipe.build()
        self.loader = DALIGenericIterator(
            pipelines=[self.pipe],
            output_map=["data", "label", "frame_idx", "start_time"],
            size=self.pipe.epoch_size("Reader"),
        )

    def __len__(self):
        return self.pipe.epoch_size("Reader")

    def __next__(self):
        return next(self.loader)

    def __iter__(self):
        return self
