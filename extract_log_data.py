import argparse
import glob
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


def log_parser(log_path):
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
    return loader_types, loaders_hz, no_workers, frame_rate

def get_gpu(gpu_file_path):
    with open(gpu_file_path, "r") as f:
        gpu_file = f.read().splitlines()
    gpu_used = (' ').join(gpu_file[7].split()[2:4])
    return gpu_used

def get_dimensions(video_info_path):
    with open(video_info_path, "r") as f:
        video_info = f.read()
    regexp = r",\s(\d+)x(\d+)"
    frame_dimension = re.search(regexp, video_info)[0][2:]
    return frame_dimension

def update_readme(readme_path, gpu_used, loader_types, loaders_avg,
no_workers, frame_rate, frame_dimension, readme_dest):
    with open(readme_path, "r") as f:
        readme_file = f.read().splitlines()
    patterns = [loader_type.capitalize() for loader_type in loader_types]
    data = []
    for row in readme_file:
        new_row = None
        for pattern in patterns:
            tokens = row.split(f"{pattern}")
            if len(tokens) == 2:
                if "|" in tokens[1]:
                    value = tokens[1].split("|")[1]
                    new_row = row.replace(value, str(round(
                        statistics.mean(loaders_avg[pattern.lower()]['clip hz: (avg)']),
                        4)))
        if len(row.split("GPU")) == 2:
            new_row = f"* GPU - {gpu_used}"
        if len(row.split("CPU")) == 2:
            new_row = f"* Number of CPU workers - {no_workers}"
        if len(row.split("Frame rate")) == 2:
            new_row = f"* Frame rate - {frame_rate} fps"
        if len(row.split("Dimensions")) == 2:
            new_row = f"* Dimensions - {frame_dimension}"
        if new_row is None:
            data.append(row)
        else:
            data.append(new_row)


    with open(readme_dest, "w") as f:
        f.write("\n".join(data))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--log_dir", default="data/logs")
    args = args.parse_args()

    current_path = Path(args.log_dir)
    results_path = current_path.parent.parent / "README.md"
    readme_dest = current_path.parent.parent / "READMEnew.md"
    list_of_files = glob.glob(str(current_path / "*.txt"))
    latest_file = max(list_of_files, key=os.path.getctime)
    loader_types, loaders_hz, no_workers, frame_rate = log_parser(latest_file)
    gpu_used = get_gpu(current_path.parent.parent / "info.txt")
    video_info_path = glob.glob(str(current_path  / "ffprobe*.log"))[0]
    frame_dimension = get_dimensions(video_info_path)

    #clips_hz_avg = statistics.mean(hz['clip hz: (avg)'])
    update_readme(results_path, gpu_used, loader_types,
     loaders_hz, no_workers, frame_rate, frame_dimension, readme_dest)

if __name__ == "__main__":
    main()
