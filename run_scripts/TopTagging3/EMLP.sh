gpu=0
for group in SO13p SO13 Lorentz; do
    for N in 100 316 1000; do
        for seed in 0 1 42; do
            echo "TopTagging3 EMLP $group N:$N seed:$seed"
            python experiment.py --dataset TopTagging --N $N --network EMLP --group $group --width 200 200 200 --steps 1000 --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
        done
    done
    for N in 3162 10000; do
        for seed in 0 1 42; do
            echo "TopTagging3 EMLP $group N:$N seed:$seed"
            python experiment.py --dataset TopTagging --N $N --network EMLP --group $group --width 200 200 200 --steps 2000 --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
        done
    done
done
