uv run main.py --mode train --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv \
  --name cctv \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uv run main.py --mode train --gpu_ids "1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam \
  --name dashcam \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uv run main.py --mode train --gpu_ids "0,1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual \
  --name dual \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/test/blur \
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
uv run inference.py \
-c checkpoints/dual \
-i /mnt/data/nblong-t04/LPDGAN/dataset/teams/test/blur \
-o results/dual/inference/teams 

uv run main.py --mode test \
--epoch 95 \
--name cctv \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/teams 
uv run main.py --mode test \
--epoch 95 \
--name dashcam \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/teams 

uv run main.py --mode train --gpu_ids "0,1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual \
  --name dual \
  --batch_size 40 \
  --num_worker 8 \
  --num_threads 8 

uv run main.py --mode test \
--load_iter 95 \
--name dual \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv
uv run main.py --mode test \
--load_iter 95 \
--name dual \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam
uv run main.py --mode test \
--load_iter 95 \
--name dual \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual

uv run main.py --mode test \
--load_iter 10 \
--name LPDGAN \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv

uv run main.py --mode test \
--load_iter 10 \
--name LPDGAN \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/teams

uv run main.py --mode train --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual \
  --name dual \
  --batch_size 40 \
  --num_worker 8 \
  --num_threads 8 

uv run main.py \
  --mode train \
  --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur \
  --name LPBlur \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8 

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/blur \
-o results/LPBlur/inference/LPBlur \
-c checkpoints/LPBlur

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/test/blur \
-o results/LPBlur/inference/LPBlur \
-c checkpoints/LPBlur

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/teams/test/blur \
-o results/LPBlur/inference/team \
-c checkpoints/LPBlur

uv run main.py --mode test \
--name LPBlur \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/teams

uv run main.py --mode test \
--name LPBlur \
--dataroot /mnt/data/nblong-t04/LPDGAN/dataset/adnl

uv run inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/teams/test/blur \
-o results/LPBlur/inference/team \
-c checkpoints/LPBlur

# train on tnadmin
$logDir = "G:\nblongT04\LPDGAN\logs_train"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$p = Start-Process uv -ArgumentList @(
    "run", "main.py"
) -RedirectStandardOutput "$logDir\train.log" `
  -RedirectStandardError "$logDir\train.err" `
  -PassThru

$p.Id | Out-File "$logDir\train.pid"

uv run main.py --mode train --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp \
  --name quan_lp --batch_size 16 \
  --num_worker 8 --num_threads 8 2>&1 | tee train-quanlp-200.txt


