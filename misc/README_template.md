## Comparing video loaders
This tool compares videoloaders for pytorch:
* Frame loading
* [Videoclip library](https://github.com/pytorch/vision/blob/master/torchvision/datasets/video_utils.py)
* [Dali loader](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html)

### Benchmark
| Loader | Clips (Hz) |
| ------ | ---------- |
| Frames | {{frames_hz}}          |
| Videoclip |  {{videoclip_hz}}      |
| Dali | {{dali_hz}} |

### Machine settings
* GPU - {{gpu_type}}
* Number of CPU workers - {{cpu_workers}}
* RAM disk
* Local SSD
* Beegfs

### Data
* Videos
* Dimensions - {{video_dims}} 
* Frame rate - {{frame_rate}} fps

### Model
* R(2+1)d with 18 layers - r2plus1d_18