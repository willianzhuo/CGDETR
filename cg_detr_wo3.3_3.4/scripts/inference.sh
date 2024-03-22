ckpt_path=$1
eval_split_name=$2
eval_path=data/highlight_${eval_split_name}_release.jsonl
# eval_path=data/tacos/${eval_split_name}.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python cg_detr_wo3.3_3.4/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
