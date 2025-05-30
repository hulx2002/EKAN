gpu=0
for width in 111 153 209 283 383; do
    for seed in 0 1 42; do
        echo "3Body MLP width:$width seed:$seed"
        python test_equi.py --dataset 3Body --network MLP --group SO2 --width $width $width $width --batch 500 --gpu $gpu --seed $seed
    done
done
for width in 111 153 209 283 383; do
    for seed in 0 1 42; do
        echo "3Body MLP augmentation width:$width seed:$seed"
        python test_equi.py --dataset 3Body --network MLP --group SO2 --width $width $width $width --batch 500 --augmentation --gpu $gpu --seed $seed
    done
done
