gpu=0
for group in SO2 O2; do
    for width in 45 88 151 262 457; do
        for seed in 0 1 42; do
            echo "3Body EKAN $group width:$width seed:$seed"
            python experiment.py --dataset 3Body --network EKAN --group $group --width $width --grid 3 --steps 5000 --update_grid --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
        done
    done
done
