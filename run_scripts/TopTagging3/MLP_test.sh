gpu=0
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging3 MLP N:$N seed:$seed"
        python test.py --dataset TopTagging --N $N --network MLP --width 200 200 200 --batch 500 --gpu $gpu --seed $seed
    done
done
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging3 MLP augmentation N:$N seed:$seed"
        python test.py --dataset TopTagging --N $N --network MLP --width 200 200 200 --batch 500 --augmentation --gpu $gpu --seed $seed
    done
done