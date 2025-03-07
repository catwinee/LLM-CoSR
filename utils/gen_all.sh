rates=(70 80 90)

for rate in "${rates[@]}"; do
    echo "Running experiment with rate: $rate"
    python temp/augment.py $rate
done

echo "All experiments completed!"