The code is derived from https://github.com/facebookresearch/fairseq

The main changes are in fairseq2/fairseq/criterions/mrt_mlm.py and mrt_bleu.py

Dependencies
https://github.com/PrithivirajDamodaran/Gramformer
https://github.com/awslabs/mlm-scoring

Usage:
1. Make sure necessary pytorch with compatiable version and other dependencies are installed
Install the modified fairseq library with 
  pip install -e .

Use standard commands from https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train
Use --criterion mrt_mlm to use our method.

## Pretrained Models
1. Simple Baseline: Follow the steps from: https://github.com/babangain/english_hindi_translation
2. Robust Baseline:
Copy the combined files to data/samanantar.noise and perform deduplication and add noise operations with dedup.py and add_noise.py
```
DATA_FOLDER_NAME=samanantar.noise
DATA_DIR=data/$DATA_FOLDER_NAME
MOSES_DIR=mosesdecoder
cat $DATA_DIR/train.dedup.en | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/train.lc.en
cat $DATA_DIR/train.dedup.hi | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/train.lc.hi

cp en_hi/bpecode $DATA_DIR/bpecode
cp en_hi/vocab.en $DATA_DIR/vocab.en

cat $DATA_DIR/train.lc.en $DATA_DIR/train.noise.en > $DATA_DIR/train.combined.en
cat $DATA_DIR/train.lc.hi $DATA_DIR/train.lc.hi > $DATA_DIR/train.combined.hi


FASTBPE_DIR=fastBPE
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.en $DATA_DIR/train.combined.en $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.hi $DATA_DIR/train.combined.hi $DATA_DIR/bpecode

BINARY_DATA_DIR=data_bin/$DATA_FOLDER_NAME
mkdir -p $BINARY_DATA_DIR
fairseq-preprocess \
    --source-lang en --target-lang hi \
    --joined-dictionary \
    --srcdict $DATA_DIR/vocab.en \
    --trainpref $DATA_DIR/train.bpe \
    --destdir $BINARY_DATA_DIR \
    --workers 20


MODEL_DIR=models/$DATA_FOLDER_NAME
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=5,6
nohup fairseq-train --fp16 \
    $BINARY_DATA_DIR \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 --disable-validation --valid-subset train \
    --max-tokens 4000 --update-freq 64  \
    --max-epoch 30 \
    --save-interval 5 \
    --save-dir $MODEL_DIR > $DATA_DIR/baseline.log &
```

```
cd fairseq2
DATA_FOLDER_NAME=flipkart_questions_50k
DATA_DIR=../data/$DATA_FOLDER_NAME
BINARY_DATA_DIR=../data_bin/$DATA_FOLDER_NAME
export CUDA_VISIBLE_DEVICES=3
```
Training with MLE
```
MODEL_DIR=../models/fk_questions_50k_finetune_batch_200tok_mle
nohup python -u train.py $BINARY_DATA_DIR --memory-efficient-fp16 \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --lr 0.000005 \
    --task translation \
    --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr-scheduler inverse_sqrt \
    --max-tokens 200 --update-freq 1 \
    --max-update 5000 \
    --save-interval 1 --save-interval-updates 500\
    --patience 5 \
    --disable-validation \
    --finetune-from-model ../models/samanantar.new/checkpoint_last.pt \
    --save-dir $MODEL_DIR > $DATA_DIR/fk_questions_50k_finetune_batch_200tok_mle.out &

OUTFILENAME=$DATA_DIR/fk_questions_50k_finetune_batch_200tok_mle_valid
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path $MODEL_DIR/checkpoint_1_200.pt  --remove-bpe  --gen-subset valid \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/valid.hi  -m bleu ter

```
Replace 2_600 with all other checkpoints and observe the one with highest BLEU score. Then generate from that checkpoint on test set
For MLE, it is checkpoint_1_2000.pt
```
OUTFILENAME=$DATA_DIR/fk_questions_50k_finetune_batch_200tok_mle_test
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path $MODEL_DIR/checkpoint_1_2000.pt  --remove-bpe  --gen-subset test \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/valid.hi  -m bleu ter
```
Training with MRT BLEU
```
MODEL_DIR=../models/$DATA_FOLDER_NAME.lr_0.000005_batch_200_bleu
export CUDA_VISIBLE_DEVICES=2
nohup python -u train.py $BINARY_DATA_DIR --memory-efficient-fp16 \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --lr 0.000005 \
    --task translation \
    --criterion mrt_bleu \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr-scheduler inverse_sqrt \
    --max-tokens 200 --update-freq 1 \
    --max-update 5000 \
    --save-interval-updates 250 \
    --patience 5 \
    --disable-validation \
    --finetune-from-model ../models/samanantar.new/checkpoint_last.pt \
    --save-dir $MODEL_DIR > $DATA_DIR/$DATA_FOLDER_NAME.lr_0.000005_batch_200_bleu.out &
```

Training with MRT MLM & BERTScore
```
MODEL_DIR=../models/fk_questions_50k_gramformer_and_orig_mrt_div_mle_tokens_beam5
nohup python -u train.py $BINARY_DATA_DIR --memory-efficient-fp16 \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --lr 0.000005 \
    --task translation \
    --criterion mrt_mlm \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr-scheduler inverse_sqrt \
    --max-tokens 200 --update-freq 1 \
    --max-update 5000 \
    --save-interval 1 --save-interval-updates 500\
    --patience 5 \
    --disable-validation \
    --finetune-from-model ../models/samanantar.new/checkpoint_last.pt \
    --save-dir $MODEL_DIR > $DATA_DIR/fk_questions_50k_gramformer_and_orig_mrt_div_mle_tokens_beam5.out &
```

# Generate with valid subset
OUTFILENAME=$DATA_DIR/flipkart_questions_50k_mbart_0.85_mlm0.15_qs_sep_grammar_corrected_beam1_best_gformer_best_of_three_lr_0.00005_2_600.valid
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path $MODEL_DIR/checkpoint_2_600.pt  --remove-bpe  --gen-subset valid \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/valid.hi  -m bleu ter

Use DATA_FOLDER_NAME=flipkart_questions_50k_new and --finetune-from-model ../models/samanantar.noise/checkpoint15.pt for fine-tuning with robust baseline

For GUDA, Refer to https://github.com/trangvu/guda