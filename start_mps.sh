export CUDA_VISIBLE_DEVICES="0"
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d