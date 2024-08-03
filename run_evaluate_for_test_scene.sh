source /home/siyanlinux/.bashrc
cd /home/siyanlinux/Documents/gaussian-splatting
conda activate py309gs

scenes=("fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
# scenes=("KingsCollege" "OldHospital" "ShopFacade" "StMarysChurch")


# for scn in "${scenes[@]}"
# do

#     test_scene_path="scene_${scn}/test_100_byorder_35"

#     if python evaluate_dof.py -s ${test_scene_path} -m scene_"${scn}"/train_50_byorder_99/output_setting1 --resolution 1;
#     then
#         echo "[Success 1] train scene_" + "${scn}" + " 50_99"
#     else
#         echo "[Failed 1] training process for scene_" + "${scn}" + " 50_99"
#     fi

#     if python evaluate_dof.py -s ${test_scene_path} -m scene_"${scn}"/train_100_byorder_90/output_setting1 --resolution 1;
#     then
#         echo "[Success 1] train scene_" + "${scn}" + " 100_90"
#     else
#         echo "[Failed 1] training process for scene_" + "${scn}" + " 100_90"
#     fi

#     if python evaluate_dof.py -s ${test_scene_path} -m scene_"${scn}"/train_200_byorder_20/output_setting1 --resolution 1;
#     then
#         echo "[Success 2] train scene_" + "${scn}" + "200_20"
#     else
#         echo "[Failed 2] training process for scene_" + "${scn}" + "200_20"
#     fi

#     # if python evaluate_dof.py -s scene_"${scn}"/train_full_byorder_85 -m scene_"${scn}"/train_full_byorder_85/output_setting1 --resolution 1;
#     # then
#     #     echo "[Success 2] train scene_" + "${scn}" + "full_85"
#     # else
#     #     echo "[Failed 2] training process for scene_" + "${scn}" + "full_85"
#     # fi
# done


scenes_camb=("KingsCollege" "OldHospital" "ShopFacade" "StMarysChurch")

for scn in "${scenes_camb[@]}"
do

    if python evaluate_dof.py -s scene_"${scn}"/train_50_byorder_99 -m scene_"${scn}"/train_50_byorder_99/output_setting1 --resolution 1;
    then
        echo "[Success 1] train scene_" + "${scn}" + " 50_99"
    else
        echo "[Failed 1] training process for scene_" + "${scn}" + " 50_99"
    fi

    if python evaluate_dof.py -s scene_"${scn}"/train_100_byorder_90 -m scene_"${scn}"/train_100_byorder_90/output_setting1 --resolution 1;
    then
        echo "[Success 1] train scene_" + "${scn}" + " 100_90"
    else
        echo "[Failed 1] training process for scene_" + "${scn}" + " 100_90"
    fi

    # if python evaluate_dof.py -s scene_"${scn}"/train_full_byorder_85 -m scene_"${scn}"/train_full_byorder_85/output_setting1 --resolution 1;
    # then
    #     echo "[Success 2] train scene_" + "${scn}" + "full_85"
    # else
    #     echo "[Failed 2] training process for scene_" + "${scn}" + "full_85"
    # fi
done