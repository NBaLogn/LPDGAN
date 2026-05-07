uvr main.py --mode train --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv \
  --name cctv \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uvr main.py --mode train --gpu_ids "1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam \
  --name dashcam \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uvr main.py --mode train --gpu_ids "0,1" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual \
  --name dual \
  --batch_size 16 \
  --num_worker 8 \
  --num_threads 8

uvr inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv/test/blur \
-o results/cctv/inference \
-c checkpoints/cctv

uvr inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/blurry-plates \
-o results/dashcam/inference/blurry-plates \
-c checkpoints/dashcam

uvr inference.py \
-i /mnt/data/nblong-t04/LPDGAN/dataset/blurry-plates \
-o results/dual/inference/blurry-plates \
-c checkpoints/dual

uvr main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv 
uvr main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv 
# uvr main.py --mode test \
#   --name dual \
#   --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv 


uvr main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam 
uvr main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam 
# uvr main.py --mode test \
#   --name dual \
#   --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam


uvr main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual 
uvr main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual 
# uvr main.py --mode test \
#   --name dual \
#   --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual 

uvr inference.py \
-c checkpoints/cctv \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/test/blur \
-o results/cctv/inference/dash

uvr inference.py \
-c checkpoints/cctv \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/test/blur \
-o results/cctv/inference/dashcam
uvr inference.py \
-c checkpoints/dashcam \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/test/blur \
-o results/dashcam/inference/cctv 

uvr inference.py \
-c checkpoints/cctv \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/test/blur \
-o results/cctv/inference/cctv
uvr inference.py \
-c checkpoints/dashcam \
-i /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/test/blur \
-o results/dashcam/inference/dashcam 


uvr main.py --mode test \
  --name cctv \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/blurry-plates 
uvr main.py --mode test \
  --name dashcam \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/blurry-plates 