export CUDA_VISIBLE_DEVICES=0,1,2
python ../src_code/run_esim.py \
--data_dir ../../dataset/ \
--do_train \
--do_eval \
--learning_rate 5e-4 \
--query_maxlen 85 \
--output_dir ../outputs/ \
--per_gpu_eval_batch_size=128   \
--per_gpu_train_batch_size=128   \
--embeddings_file ../../dataset/sgns.wiki.bigram-char \
--train_language zh \
--train_steps 1070 \
--eval_steps 105