cd /home/siyanlinux/Documents/datasets/HongKong/full_arcore/colmap_rebuild_sfm/square/

mkdir sparse/merge

colmap model_merger --input_path1 sparse/1 --input_path2 sparse/0 --output_path sparse/merge
#--image_path /images

cd /home/siyanlinux/Documents/gaussian-splatting