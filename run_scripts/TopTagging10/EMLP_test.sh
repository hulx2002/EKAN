gpu=0
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging10 EMLP SO13p N:$N seed:$seed"
        python test.py --dataset TopTagging --n_component 10 --N $N --network EMLP --group SO13p --width 200 200 200 --batch 500 --gpu $gpu --seed $seed
    done
done
