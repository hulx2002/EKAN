gpu=0
for width in 76 134 238 423 752; do
    for seed in 0 1 42; do
        echo "3Body KAN width:$width seed:$seed"
        python test_equi.py --dataset 3Body --network KAN --group SO2 --width $width --grid 3 --batch 500 --gpu $gpu --seed $seed
    done
done
for width in 76 134 238 423 752; do
    for seed in 0 1 42; do
        echo "3Body KAN augmentation width:$width seed:$seed"
        python test_equi.py --dataset 3Body --network KAN --group SO2 --width $width --grid 3 --batch 500 --augmentation --gpu $gpu --seed $seed
    done
done
