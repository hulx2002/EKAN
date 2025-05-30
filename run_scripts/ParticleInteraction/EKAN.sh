gpu=0
for group in SO13p SO13 Lorentz; do
    for N in 100 316; do
        for seed in 0 1 42; do
            echo "ParticleInteraction EKAN $group N:$N seed:$seed"
            python experiment.py --dataset ParticleInteraction --N $N --network EKAN --group $group --width 1000 --grid 3 --steps 7000 --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
        done
    done
    for N in 1000 3162 10000; do
        for seed in 0 1 42; do
            echo "ParticleInteraction EKAN $group N:$N seed:$seed"
            python experiment.py --dataset ParticleInteraction --N $N --network EKAN --group $group --width 1000 --grid 3 --steps 15000 --lr 3e-3 --batch 500 --gpu $gpu --seed $seed
        done
    done
done
