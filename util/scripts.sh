mv test/blur_cctv cctv/test/blur
mv train/blur_cctv cctv/train/blur
mv val/blur_cctv cctv/val/blur

mv test/blur_dashcam dashcam/test/blur
mv train/blur_dashcam dashcam/train/blur
mv val/blur_dashcam dashcam/val/blur

mv test/blur_dual dual/test/blur
mv train/blur_dual dual/train/blurr
mv val/blur_dual dual/val/blur



ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/test/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/train/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/val/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/cctv/val/sharp

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/test/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/train/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/val/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dashcam/val/sharp

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/test/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/train/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/val/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp_dataset/dual/val/sharp
