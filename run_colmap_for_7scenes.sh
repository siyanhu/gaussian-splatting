scenes=("fire" "chess" "heads" "pumpkin" "redkitchen" "stairs" "office")

categories=("train_full_byorder_85")

for scn in "${scenes[@]}"
do
    
    for cate in "${categories[@]}"
    do
        rm -rf scene_"${scn}"/"${cate}"/dense
        
        mkdir scene_"${scn}"/"${cate}"/dense

        mkdir scene_"${scn}"/"${cate}"/dense/workspace

        colmap image_undistorter --image_path scene_"${scn}"/"${cate}"/images \
        --input_path scene_"${scn}"/"${cate}"/sparse/0 \
        --output_path scene_"${scn}"/"${cate}"/dense/workspace \
        --output_type COLMAP

        colmap patch_match_stereo --workspace_path scene_"${scn}"/"${cate}"/dense/workspace \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true

        colmap stereo_fusion --workspace_path scene_"${scn}"/"${cate}"/dense/workspace --output_path scene_"${scn}"/"${cate}"/dense/workspace/fused.ply

        colmap model_converter --input_path scene_"${scn}"/"${cate}"/dense/workspace/sparse/0 \
        --output_path scene_"${scn}"/"${cate}"/dense/workspace/sparse/0 \
        --output_type TXT
    
    done
done

cd /home/siyanlinux/Documents/gaussian-splatting