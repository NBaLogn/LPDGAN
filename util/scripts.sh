uv run main.py --mode train --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv \
  --name cctv \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uv run main.py --mode train --gpu_ids "1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam \
  --name dashcam \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uv run main.py --mode train --gpu_ids "0,1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual \
  --name dual \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv/test/blur \
-o results/cctv/inference \
-c checkpoints/cctv

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/blurry-plates \
-o results/dashcam/inference/blurry-plates \
-c checkpoints/dashcam

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/blurry-plates \
-o results/dual/inference/blurry-plates \
-c checkpoints/dual

uv run main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv 
uv run main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv 
uv run main.py --mode test \
  --name dual \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv 


uv run main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam 
uv run main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam 
uv run main.py --mode test \
  --name dual \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam


uv run main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual 
uv run main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual 
uv run main.py --mode test \
  --name dual \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual 


# cross test cctv on dashcam
uv run inference.py \
-c checkpoints/cctv \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/test/blur \
-o results/cctv/inference/dashcam
# cross test dashcam on cctv
uv run inference.py \
-c checkpoints/dashcam \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/test/blur \
-o results/dashcam/inference/cctv 

# test inference on teams dataset
uv run inference.py \
-c checkpoints/cctv \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/teams/test/blur \
-o results/cctv/inference/teams
uv run inference.py \
-c checkpoints/dashcam \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/teams/test/blur \
-o results/dashcam/inference/teams 

uv run inference.py \
-c checkpoints/cctv \
-i /mnt/data/nblong-t04/LPDGAN/dataset/teams/test/blur \
-o results/cctv/inference/teams
uv run inference.py \
-c checkpoints/dashcam \
-i /mnt/data/nblong-t04/LPDGAN/dataset/teams/test/blur \
-o results/dashcam/inference/teams 

uv run main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/teams 
uv run main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/teams 