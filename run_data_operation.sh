scenes=("atrium" "bar" "church" "concourse" "square" "stairs_hkust")

for scn in "${scenes[@]}"
do
    cp -r ../data/scene_${scn} scene_${scn}
done
