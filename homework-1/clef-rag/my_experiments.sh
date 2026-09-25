weight_pairs=(
    "0.3 0.7"
    "0.5 0.5"
    "0.7 0.3"
)

candidate_amounts=(10 20 30)
top_ks=(3 6 9)

for amount in "${candidate_amounts[@]}"; do
    echo "Running with candidate amount: $amount"
    for k in "${top_ks[@]}"; do
        echo "Running with top_k: $k"
        for pair in "${weight_pairs[@]}"; do
            read -r w1 w2 <<< "$pair"
            echo "Running with weights: $w1, $w2"
            uv run clef-rag evaluate \
                --max_candidates "$amount" \
                --top_k "$k" \
                --retriever-weights "$w1" "$w2" \
                --output "index/evaluation_max-candidate-${amount}_top-k-${k}_weights-${w1}-${w2}.json"
        done
    done
done