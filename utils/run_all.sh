# # experiments=("misr" "fisr" "GSAT" "lightgcl")
# experiments=("misr")
# seeds=(1 12 123 1234 12345)

# for exper in "${experiments[@]}"; do
#     for seed in "${seeds[@]}"; do
#         echo "Running experiment: $exper with seed: $seed"
#         python train.py experiment="$exper" ++seed="$seed"
#     done
# done

# echo "All experiments completed!"
rates=(10 20 30 40 50 60 70 80 90)
seeds=(1 12 123 1234 12345)

for rate in "${rates[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running experiment with rate: $rate"
        ratio=$(printf "%.2f" $(echo "$rate/100" | bc -l))
        python train.py experiment=contras-mm \
            ++model.view_name="views-${rate}%-20.pkl" \
            ++seed=$seed \
            ++logger.wandb.name="Contras-MM-$rate-Minimal-20"
    done
done

echo "All experiments completed!"