gpu=0
for group in SO13p SO13 Lorentz; do
    for N in 100 316 1000 3162 10000; do
        for seed in 0 1 42; do
            echo "TopTagging3 EKAN $group N:$N seed:$seed"
            python test.py --dataset TopTagging --N $N --network EKAN --group $group --width 200 --grid 3 --batch 500 --gpu $gpu --seed $seed
        done
    done
done
