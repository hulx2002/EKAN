gpu=0
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "TopTagging3 CGENN N:$N seed:$seed"
        python test.py --dataset TopTagging --N $N --hidden_features 200 --n_layers 4 --batch 500 --gpu $gpu --seed $seed
    done
done
