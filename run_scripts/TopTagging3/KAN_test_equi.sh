gpu=0
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging3 KAN N:$N seed:$seed"
        python test_equi.py --dataset TopTagging --N $N --network KAN --group SO13p --width 384 --grid 3 --batch 500 --gpu $gpu --seed $seed
    done
done
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging3 KAN augmentation N:$N seed:$seed"
        python test_equi.py --dataset TopTagging --N $N --network KAN --group SO13p --width 384 --grid 3 --batch 500 --augmentation --gpu $gpu --seed $seed
    done
done