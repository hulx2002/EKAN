gpu=0
for group in SO2 O2; do
    for width in 84 110 147 214 281; do
        for seed in 0 1 42; do
            echo "3Body EMLP $group width:$width seed:$seed"
            python experiment.py --dataset 3Body --network EMLP --group $group --width $width $width $width --steps 5000 --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
        done
    done
done
