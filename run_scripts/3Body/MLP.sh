gpu=0
for width in 111 153 209 283 383; do
    for seed in 0 1 42; do
        echo "3Body MLP width:$width seed:$seed"
        python experiment.py --dataset 3Body --network MLP --width $width $width $width --steps 5000 --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
    done
done
for width in 111 153 209 283 383; do
    for seed in 0 1 42; do
        echo "3Body MLP augmentation width:$width seed:$seed"
        python experiment.py --dataset 3Body --network MLP --group SO2 --width $width $width $width --steps 5000 --lr 3e-3 --batch 500 --augmentation --gpu $gpu --seed $seed
    done
done
