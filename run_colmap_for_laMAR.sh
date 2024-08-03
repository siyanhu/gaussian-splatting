cd /home/siyanlinux/Documents/LaMAR/colmap/HGE/query_val_phone/ios_2022-01-25_14/

mkdir dense

mkdir dense/workspace

colmap feature_extractor --database_path dense/database.db --image_path images

colmap exhaustive_matcher --database_path dense/database.db

colmap point_triangulator --database_path dense/database.db --image_path images --input_path sparse/preset --output_path sparse

colmap image_undistorter --input_path sparse/0 --output_path dense/workspace

colmap patch_match_stereo --workspace_path dense/workspace

colmap stereo_fusion --workspace_path dense/workspace --output_path dense/workspace/fused.ply

cd /home/siyanlinux/Documents/gaussian-splatting