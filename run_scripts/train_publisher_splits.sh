test_publisher_name="$1"
lr=$2
model_name="$3"
fewshot_lr=$4
seed=$5

num_fewshot_epochs=250
episode_num=0

cd ../tools

n_gpu=1
num_train_epochs=10
eval_per_epoch=4
batch_size=4
meta_model_type="ivila"
agg_level="block"
used_token="BLK"

echo "Test Publisher Name:                      $test_publisher_name"
echo "Model Name:                              $model_name"
echo "Training Epochs:                         $num_train_epochs"
echo "The group level is:                      $agg_level"
echo "The used layout indicator token is:      $used_token"

output_dir="../checkpoints/grotoap2/${model_name//\//-}_${test_publisher_name}_lr_${lr}_seed_${seed}"
echo "The results will be saved in '$output_dir'"

python train-model.py \
  --model_name_or_path "$model_name" \
  --test_publisher_name "$test_publisher_name" \
  --preprocessing_num_workers 20 \
  --output_dir "$output_dir" \
  --do_eval_predictions \
  --save_strategy 'epoch' \
  --metric_for_best_model 'fscore' \
  --evaluation_strategy 'epoch' \
  --num_train_epochs $num_train_epochs \
  --save_total_limit 2 \
  --per_device_train_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --warmup_steps 2000 \
  --load_best_model_at_end \
  --added_special_sepration_token $used_token \
  --agg_level $agg_level \
  --fp16 \
  --use_auth_token \
  --not_resume_training \
  --overwrite_output_dir \
  --learning_rate $lr \
  --overwrite_cache \
  --do_predict \
  --do_train \
  --num_fewshot_epochs $num_fewshot_epochs \
  --fewshot_episode_num $episode_num \
  --fewshot_lr $fewshot_lr \
  --seed $seed
