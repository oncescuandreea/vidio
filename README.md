## Comparing video loaders
This tool compares videoloaders for pytorch:
* Frame loading
* [Videoclip library](https://github.com/pytorch/vision/blob/master/torchvision/datasets/video_utils.py)
* [Dali loader](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html)

### Benchmark
| Loader | Clips (Hz) |
| ------ | ---------- |
| Frames | 18.1          |
| Videoclip |  20.3      |
| Dali | 44.6 |

### Machine settings
* GPU - GeForce GTX
* Number of CPU workers - 8
* RAM disk
* Local SSD
* Beegfs

### Data
* Videos
* Dimensions - height: 436, width: 1024 
* Frame rate - 24  fps

### Model
* R(2+1)d with 18 layers - r2plus1d_18
