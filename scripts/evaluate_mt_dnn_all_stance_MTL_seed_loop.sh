#!/bin/bash
if [[ $# -ne 7 ]]; then
  echo "bash evaluate_mt_dnn_all_stance_MTL_seed_loop.sh <gpu> <model_name> <basedir> <timestamp> <train_data_ratio> <lambda> <connectedlayer>"
  echo "<train_data_ratio> in percentage (i.e. 10, 20, 30, ...)!"
  exit 1
fi

data="arc,argmin,fnc1,ibmcs,iac1,perspectrum,semeval2016t6,semeval2019t7,scd,snopes"
seeds=("0" "1" "2" "3" "4")
model=$2 #bert_model_large, mt_dnn_large

prefix="mt-dnn-${data}_ST"
BATCH_SIZE=16
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
basedir=$3
tstr=$4 # e.g. "2019-06-19T1750"

train_datasets=${data}
test_datasets=${data}
stress_tests="negation,spelling,paraphrase"
DATA_DIR="${basedir}/data/mt_dnn"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
epochs=5
max_seq_len=100 # for the longer stress tests it will automatically choose 512
dump_to_checkpoints=1 # if 0, only dumps results to result folder
train_data_ratio=$5
lambda=$6
connectedlayer=$7

for seed in "${seeds[@]}" ; do
    if [[ $train_data_ratio -eq 100 ]]; then
        model_dir="${basedir}/debias_checkpoints/${prefix}_seed${seed}_ep${epochs}_${model}_answer_opt${answer_opt}_lambda${lambda}_connectedlayer${connectedlayer}_${tstr}"
    else
        model_dir="${basedir}/debias_checkpoints/${prefix}_seed${seed}_ep${epochs}_${model}_answer_opt${answer_opt}_trainratio${train_data_ratio}_lambda${lambda}connectedlayer${connectedlayer}_${tstr}"
    fi
    echo $model_dir
    BERT_PATH="${model_dir}/model.pt"
    cp "../mt_dnn_models/vocab.txt" $model_dir
    log_file="${model_dir}/log.log"
    python -W ignore ../predict.py --train_data_ratio ${train_data_ratio} --dump_to_checkpoints ${dump_to_checkpoints} --stress_tests ${stress_tests} --max_seq_len ${max_seq_len} --seed ${seed} --epochs ${epochs} --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${test_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --dump_representations --connectedlayer ${connectedlayer}
done
