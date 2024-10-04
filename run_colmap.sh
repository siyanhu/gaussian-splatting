scenes=("atrium" "bar" "church" "concourse" "office" "square" "stairs")

for scn in "${scenes[@]}"
do
    mkdir "${scn}"/dense

    mkdir "${scn}"/sparse

    colmap feature_extractor --database_path "${scn}"/dense/database.db --image_path "${scn}"/images 

    colmap exhaustive_matcher --database_path "${scn}"/dense/database.db

    colmap mapper --database_path "${scn}"/dense/database.db --image_path "${scn}"/images --output_path "${scn}"/sparse

    colmap model_converter --input_path "${scn}"/sparse/0 --output_path "${scn}"/sparse --output_type TXT
    
done

    
python train.py -s HKUST/sub1 -m HKUST/sub1/output && \
python train.py -s HKUST/sub2 -m HKUST/sub2/output && \
python train.py -s HKUST/sub3 -m HKUST/sub3/output && \
python train.py -s HKUST/sub4 -m HKUST/sub4/output && \
python train.py -s red3 -m red3/output && \
python train.py -s HKUST/sub6 -m HKUST/sub6/output