cd /home/siyanlinux/Documents/datasets/HongKong/full_arkit/colmap_rebuild_sfm/


scenes=("atrium" "bar" "church" "concourse" "office" "square" "stairs")

for scn in "${scenes[@]}"
do
    mkdir "${scn}"/dense

    mkdir "${scn}"/sparse

    colmap feature_extractor --database_path "${scn}"/dense/database.db --image_path "${scn}"/images 

    colmap sequential_matcher --database_path "${scn}"/dense/database.db

    colmap mapper --database_path "${scn}"/dense/database.db --image_path "${scn}"/images --output_path "${scn}"/sparse

    colmap model_converter --input_path "${scn}"/sparse/0 --output_path "${scn}"/sparse/0 --output_type TXT

    rm -rf "${scn}"/dense

    mkdir "${scn}"/dense
    
    mkdir "${scn}"/dense/workspace

    colmap image_undistorter --image_path "${scn}"/images --input_path "${scn}"/sparse/0 --output_path "${scn}"/dense/workspace --output_type COLMAP

    colmap patch_match_stereo --workspace_path "${scn}"/dense/workspace --workspace_format COLMAP --PatchMatchStereo.geom_consistency true

    colmap stereo_fusion --workspace_path "${scn}"/dense/workspace --output_path "${scn}"/dense/workspace/fused.ply
    
done

cd /home/siyanlinux/Documents/gaussian-splatting