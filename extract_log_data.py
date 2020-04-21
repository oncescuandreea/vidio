import os
import re
import argparse
import statistics
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict

from typeguard import typechecked

from compare_loaders import get_gpu_info_path


@typechecked
def log_parser(log_dir: Path):
    list_of_files = list(log_dir.glob("*_*.txt"))
    log_path = max(list_of_files, key=os.path.getctime)
    with open(log_path, "r") as f:
        log_file = f.read().splitlines()
    hz = defaultdict(list)
    loader_types = log_file[0].split("INFO:profiler:Profiling ")[1]
    loader_types = loader_types[1:-1].split(",")
    no_loaders = len(loader_types)
    for counter in range(0, no_loaders):
        loader_types[counter] = str(loader_types[counter]).replace(' ', '')
        loader_types[counter] = loader_types[counter][1:-1]
    no_workers = log_file[1].split("Number of CPU workers - ")[1]
    frame_rate = log_file[2].split("Frame rate - ")[1].split("fps")[0]
    loaders_hz = dict.fromkeys(loader_types)
    patterns = ["batch hz: (avg)", "clip hz: (avg)"]
    for loader_type in loader_types:
        hz = defaultdict(list)
        for row in log_file:
            if len(row.split(f"{loader_type}")) == 2:
                for pattern in patterns:
                    tokens = row.split(f"{pattern}")
                    if len(tokens) == 2:
                        value = tokens[1].split(",")[0]
                        hz[pattern].append(float(value))
        loaders_hz[loader_type] = hz
    return loaders_hz, no_workers, frame_rate


@typechecked
def get_gpu_type(log_dir: Path) -> str:
    gpu_info_path = get_gpu_info_path(log_dir=log_dir)
    with open(gpu_info_path, "r") as f:
        gpu_file = f.read().splitlines()
    return " ".join(gpu_file[7].split()[2:4])


@typechecked
def get_dimensions(log_dir: Path) -> str:
    video_info_path = list(log_dir.glob("video-meta-*.log"))[0]
    with open(video_info_path, "r") as f:
        video_info = f.read().rstrip()
    return video_info


@typechecked
def update_readme(readme_template_path: Path, readme_dest_path: Path, log_dir: Path):
    gpu_type = get_gpu_type(log_dir=log_dir)
    video_dims = get_dimensions(log_dir=log_dir)
    loaders_hz, cpu_workers, frame_rate = log_parser(log_dir=log_dir)
    with open(readme_template_path, "r") as f:
        readme_file = f.read().splitlines()

    generated = []
    for row in readme_file:
        edits = []
        regex = r"\{\{(.*?)\}\}"
        for match in re.finditer(regex, row):
            groups = match.groups()
            assert len(groups) == 1, "expected single group"
            target = groups[0]
            if target == "frame_rate":
                token = frame_rate
            elif target == "video_dims":
                token = video_dims
            elif target == "gpu_type":
                token = gpu_type
            elif target == "cpu_workers":
                token = cpu_workers
            elif target in {"frames_hz", "videoclip_hz", "dali_hz"}:
                warmup = 5
                key = target.replace("_hz", "")
                stat = statistics.mean(loaders_hz[key]['clip hz: (avg)'][warmup:])
                token = f"{stat:.1f}"
            edits.append((match.span(), token))
        if edits:
            # invert the spans
            spans = [(None, 0)] + [x[0] for x in edits] + [(len(row), None)]
            inverse_spans = [(x[1], y[0]) for x, y in zip(spans, spans[1:])]
            tokens = [row[start:stop] for start, stop in inverse_spans]
            vals = [str(x[1]) for x in edits]
            new_row = ""
            for token, val in zip_longest(tokens, vals, fillvalue=""):
                new_row += token + val
            row = new_row

        generated.append(row)

    print(f"Writing updated README to {readme_dest_path}")
    with open(readme_dest_path, "w") as f:
        for row in generated:
            f.write(f"{row}\n")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--log_dir", default="data/logs", type=Path)
    args.add_argument("--readme_dest_path", default="README.md", type=Path)
    args.add_argument("--readme_template_path", type=Path,
                      default="misc/README_template.md")
    args = args.parse_args()

    update_readme(
        log_dir=args.log_dir,
        readme_dest_path=args.readme_dest_path,
        readme_template_path=args.readme_template_path,
    )


if __name__ == "__main__":
    main()
