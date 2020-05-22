#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="csl"
MODEL_NAME="roberta-large-pytorch"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export ROBERTA_WWM_LARGE_DIR=../../data/$MODEL_NAME
export GLUE_DATA_DIR=../../data/input

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

# download model
if [ ! -d $ROBERTA_WWM_LARGE_DIR ]; then
  mkdir -p $ROBERTA_WWM_LARGE_DIR
  echo "makedir $ROBERTA_WWM_LARGE_DIR"
fi
cd $ROBERTA_WWM_LARGE_DIR
if [ ! -f "bert_config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "bert_model.ckpt.index" ] || [ ! -f "bert_model.ckpt.meta" ] || [ ! -f "bert_model.ckpt.data-00000-of-00001" ]; then
  rm *
  wget -c https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
  unzip chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
  rm chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
else
  echo "model exists"
fi
echo "Finish download model."

# run task
cd $CURRENT_DIR
echo "Start running..."
python $CURRENT_DIR/cores/summary_baseline.py \
  --dict_path=$ROBERTA_WWM_LARGE_DIR/vocab.txt \
  --config_path=$ROBERTA_WWM_LARGE_DIR/bert_config.json \
  --checkpoint_path=$ROBERTA_WWM_LARGE_DIR/bert_model.ckpt \
  --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.csv \
  --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/test.csv \
  --sample_path=None \
  --albert=False \
  --epochs=10 \
  --batch_size=4 \
  --lr=1e-5 \
  --topk=1 \
  --max_input_len=256 \
  --max_output_len=32
