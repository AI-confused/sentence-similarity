export CUDA_VISIBLE_DEVICES=3
python ../src_code/run_bimpm.py \
--data_dir ../../dataset/ \
--do_train \
--do_eval \
--hidden_size 300 \
--learning_rate 1e-4 \
--query_maxlen 85 \
--output_dir ../outputs/ \
--per_gpu_eval_batch_size=128   \
--per_gpu_train_batch_size=128   \
--embeddings_file ../../dataset/sgns.wiki.bigram-char \
--train_language zh \
--train_steps 2100 \
--eval_steps 105
