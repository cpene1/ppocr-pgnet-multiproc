# ppocr-pgnet-multiproc
This is a simple script that provides an easy way to run ppocr PGNet on multiple processes. CUDA MPS is required for a reliable run.

# Usage

```
sh start_mps.sh
python pgnet_multiproc.py -num-w=2 -num-i=50
sh stop_mps.sh
```

# Acknowledgements

- [End-to-end OCR Algorithm-PGNet](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/pgnet_en.md)

# References:
- https://stackoverflow.com/questions/31643570/running-more-than-one-cuda-applications-on-one-gpu
- https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app
- https://on-demand.gputechconf.com/gtc/2015/presentation/S5584-Priyanka-Sah.pdf
- https://docs.nvidia.com/deploy/mps/index.html