# scenes=("atrium" "bar" "church" "concourse" "square" "stairs_hkust")
scenes=('red2' 'red3')

for scn in "${scenes[@]}"
do
    python train.py -s ${scn} -m ${scn}/output_3dgs
done
