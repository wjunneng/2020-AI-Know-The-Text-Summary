#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="csl"
MODEL_NAME="roberta-large-pytorch"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export ROBERTA_WWM_LARGE_DIR=/home/wjunneng/Ubuntu/2020-AI-Know-The-Text-Summary/data/$MODEL_NAME
export GLUE_DATA_DIR=/home/wjunneng/Ubuntu/2020-AI-Know-The-Text-Summary/data/output

# check python package
check_bert4keras=$(pip show bert4keras | grep "Version")

if [ ! -n "$check_bert4keras" ]; then
  pip install git+https://www.github.com/bojone/bert4keras.git@v0.3.6
else
  if [ ${check_bert4keras:8:13} = '0.3.6' ]; then
    echo "bert4keras installed."
  else
    pip install git+https://www.github.com/bojone/bert4keras.git@v0.3.6
  fi
fi

check_rouge=$(pip show rouge | grep "Version")

if [ ! -n "$check_rouge" ]; then
  pip install rouge
else
  echo "rouge installed."
fi

check_nltk=$(pip show nltk | grep "Version")

if [ ! -n "$check_nltk" ]; then
  pip install nltk
else
  echo "nltk installed."
fi

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
  --epochs=10 \
  --batch_size=4 \
  --lr=1e-5 \
  --topk=1 \
  --max_input_len=256 \
  --max_output_len=32
