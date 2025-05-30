gpu=0
for N in 100 316 1000; do
    for seed in 0 1 42; do
        echo "TopTagging20 EKAN SO13p N:$N seed:$seed"
        python experiment.py --dataset TopTagging --n_component 20 --N $N --network EKAN --group SO13p --width 200 --grid 3 --steps 1000 --update_grid --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
    done
done
for N in 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging20 EKAN SO13p N:$N seed:$seed"
        python experiment.py --dataset TopTagging --n_component 20 --N $N --network EKAN --group SO13p --width 200 --grid 3 --steps 2000 --update_grid --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
    done
done
