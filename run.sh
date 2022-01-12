# 0 bert-dense cls, fine_tuning 87 100 
for seed in 13 21 42 
do
  python prompt_tuning.py \
    --seed $seed \
    --model bert_dense \
    --mask_id  0 \
    --epochs 10 \
    --fine_tuning \
    --lr 2e-5
done


# 1 [MASK]事件： manual prompt bert-dense no mlm, fine_tuning  87 100
for seed in 13 21 42
do
  python prompt_tuning.py \
    --seed $seed \
    --manual_template [MASK]事件： \
    --model bert_dense \
    --mask_id  1 \
    --epochs 10 \
    --fine_tuning \
    --lr 2e-5
done

# 2 [MASK]事件： manual prompt bert-dense mlm, fine_tuning 87 100
for seed in 13 21 42 
do
  python prompt_tuning.py \
    --seed $seed \
    --manual_template [MASK]事件： \
    --model manual_prompt \
    --mask_id  1 \
    --epochs 10 \
    --fine_tuning \
    --lr 2e-5
done

# # 3 这是[MASK]事件。 manual prompt bert-dense no mlm, fine_tuning 87 100
# for seed in 13 21 42 
# do
#   python prompt_tuning.py \
#     --seed $seed \
#     --post \
#     --manual_template 这是[MASK]事件。 \
#     --model bert_dense \
#     --epochs 10 \
#     --fine_tuning \
#     --lr 2e-5
# done

# # 4 这是[MASK]事件。 manual prompt bert-dense mlm, fine_tuning  87 100
# for seed in 13 21 42
# do
#   python prompt_tuning.py \
#     --seed $seed \
#     --post \
#     --manual_template 这是[MASK]事件。 \
#     --model manual_prompt \
#     --epochs 10 \
#     --fine_tuning \
#     --lr 2e-5
# done

# # 5 soft prompt cls， initialize randomly, fine_tuning 87 100
# for seed in 13 21 42
# do
#     python prompt_tuning.py \
#       --seed $seed \
#       --model soft_prompt \
#       --n_tokens 2 \
#       --split1 2 \
#       --mask_id  0 \
#       --epochs 10 \
#       --fine_tuning \
#       --lr 2e-5
# done

# # 6 soft prompt mask， initialize randomly, fine_tuning  87 100
# for seed in 13 21 42
# do
#     python prompt_tuning.py \
#       --seed $seed \
#       --model soft_prompt \
#       --manual_template [MASK] \
#       --n_tokens 2 \
#       --split1 2 \
#       --mask_id  3 \
#       --epochs 10 \
#       --fine_tuning \
#       --lr 2e-5
# done


# # 7 soft prompt mlm, initialize randomly, fine_tuning p1 m p2  87 100
# for seed in 13 21 42
# do
#   python prompt_tuning.py \
#     --seed $seed \
#     --manual_template [MASK] \
#     --n_tokens 2 \
#     --split1 1 \
#     --mask_id  2 \
#     --model soft_prompt_mlm \
#     --epochs 10 \
#     --fine_tuning \
#     --lr 2e-5
# done


# 枚举
# for n in 4 6 7 8 9
# do  
#     d=$(($n+1))
#     for seed in 13 21 42 87 100
#     do
#       python prompt_tuning.py \
#         --seed $seed \
#         --manual_template [MASK] \
#         --n_tokens $n \
#         --split1 $n \
#         --mask_id  $d\
#         --model soft_prompt_mlm \
#         --epochs 10 \
#         --fine_tuning \
#         --lr 2e-5
#     done
# done
