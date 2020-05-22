#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="csl"
MODEL_NAME="roberta-large-pytorch"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export ROBERTA_WWM_LARGE_DIR=/home/wjunneng/Ubuntu/2020-AI-Know-The-Text-Summary/data/$MODEL_NAME
export GLUE_DATA_DIR=/home/wjunneng/Ubuntu/2020-AI-Know-The-Text-Summary/data/output

# run task
cd $CURRENT_DIR
echo "Start running..."
python $CURRENT_DIR/cores/summary_baseline.py \
  --dict_path=$ROBERTA_WWM_LARGE_DIR/vocab.txt \
  --config_path=$ROBERTA_WWM_LARGE_DIR/bert_config.json \
  --checkpoint_path=$ROBERTA_WWM_LARGE_DIR/bert_model.ckpt \
  --train_data_path=$GLUE_DATA_DIR/train.csv \
  --val_data_path=$GLUE_DATA_DIR/val.csv \
  --sample_path=$GLUE_DATA_DIR/sample.csv \
  --albert=False \
  --epochs=5 \
  --batch_size=4 \
  --lr=1e-5 \
  --topk=1 \
  --max_input_len=512 \
  --max_output_len=32
