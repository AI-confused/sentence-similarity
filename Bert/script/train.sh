export CUDA_VISIBLE_DEVICES=0,1,2

python ../src_code/run_bert.py \
--model_type Albert \
--model_name_or_path ~/lyl/albert_xlarge_zh/ \
--do_lower_case \
--train_language zh \
--from_tf \
--do_train \
--do_eval \
--do_test \
--data_dir ../../dataset/ \
--output_dir ../outputs/ \
--max_seq_length 170 \
--eval_steps 650 \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 16 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 10400
