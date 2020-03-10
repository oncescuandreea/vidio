## Comparing video loaders
This tool compares videoloaders for pytorch:
* Frame loading
* [Videoclip library](https://github.com/pytorch/vision/blob/master/torchvision/datasets/video_utils.py)
* [Dali loader](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html)
### Benchmark
| Loader | Clips (Hz) |
| ------ | ---------- |
| Frames | 5          |
| Videoclip |  5      |
| Dali | 16 |
### Machine settings
* GPU - Tesla P40
* Number of CPU workers - 8
* RAM disk
* Local SSD
* Beegfs
### Data
* Videos
* Dimensions - 1024x436 
* Frame rate - 24 fps

### Model
* R(2+1)d with 18 layers - r2plus1d_18