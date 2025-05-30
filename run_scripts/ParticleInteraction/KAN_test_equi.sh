gpu=0
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "ParticleInteraction KAN N:$N seed:$seed"
        python test_equi.py --dataset ParticleInteraction --N $N --network KAN --width 3840 --grid 3 --batch 500 --gpu $gpu --seed $seed
    done
done
for N in 100 316 1000 3162 10000; do
    for seed in 0 1 42; do
        echo "ParticleInteraction KAN augmentation N:$N seed:$seed"
        python test_equi.py --dataset ParticleInteraction --N $N --network KAN --width 3840 --grid 3 --batch 500 --augmentation --gpu $gpu --seed $seed
    done
done
