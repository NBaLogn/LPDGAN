mv test/blur_cctv cctv/test/blur
mv train/blur_cctv cctv/train/blur
mv val/blur_cctv cctv/val/blur

mv test/blur_dashcam dashcam/test/blur
mv train/blur_dashcam dashcam/train/blur
mv val/blur_dashcam dashcam/val/blur

mv test/blur_dual dual/test/blur
mv train/blur_dual dual/train/blurr
mv val/blur_dual dual/val/blur



ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/test/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/train/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/val/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/cctv/val/sharp

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/test/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/train/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/val/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dashcam/val/sharp

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/test/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/train/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/val/sharp /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/dual/val/sharp

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/sharp /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/train/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/blur /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/train/blur

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/sharp /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/test/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/blur /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/test/blur

ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/GT /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/sharp
ln -s /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/GT /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/blur
