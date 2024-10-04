# scenes=("fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
scenes=("atrium" "bar" "church" "concourse" "square" "stairs_hkust")
# scenes_camb=("KingsCollege" "OldHospital" "ShopFacade" "StMarysChurch")

for scn in "${scenes[@]}"
do

    if python train.py -s scene_"${scn}"/train_full_byorder_85 -m scene_"${scn}"/train_full_byorder_85/output_lod0 --resolution 1;
    then
        echo "[Success 1] train scene_" + "${scn}"
    else
        echo "[Failed 1] training process for scene_" + "${scn}"
    fi
done

python train.py -s scene_fire/train_full_byorder_85 -m scene_fire/train_full_byorder_85/output_lod0 --resolution 1
python train.py -s HKUST_Piazza  -m HKUST_Piazza/output_lod0 --resolution 1