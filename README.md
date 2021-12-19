# SparseBERT-NAACL2021

This is a reference implementation for [Rethinking Network Pruning - under the Pre-train and Fine-tune Paradigm (NAACL'21)](https://arxiv.org/pdf/2104.08682.pdf). Please feel free to contact DK Xu (dux19@psu.edu) if you have any question.

* Three sections. One is for MRPC, one is for QNLI, and the last is to print sparsity.
* In MRPC section, you will first get the eval results of the finetuned (on MRPC) BERT, then you will get the results of the provided sparse (x20) model, and finally you will compress the model and get the results of your generated sparse model.
* The details of the tasks of MRPC and QNLI can be found at <https://arxiv.org/pdf/1810.04805.pdf>
* Please follow [HuggingFace prject](https://github.com/huggingface/transformers) to construct GLUE data sets.
* Please follow [TinyBERT project](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) to construct the augmented data for each data set.


## MRPC
### Get the results (on eval set) of the finetuned BERT_base model
```
# You can download the finetuned BERT_base model at <https://huggingface.co/textattack/bert-base-uncased-MRPC>
# $/SparseBERT/finetuned_BERTs/bert_base_uncased_mrpc$ contains the finetuned BERT_base model for MRPC

TINYBERT_DIR=/SparseBERT/finetuned_BERTs/bert_base_uncased_mrpc
TASK_DIR=/SparseBERT/glue_data/MRPC
TASK_NAME=MRPC
OUTPUT_DIR=/SparseBERT/output_glue/tmp1

CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128

# Eval results: 
# acc = 0.8676470588235294
# acc_and_f1 = 0.8877484440875327
# eval_loss = 0.3441865134697694
# f1 = 0.9078498293515359
```

### Get the results (eval set) of the provided sparse (x20) BERT_base model
```
# $/SparseBERT/provided_sparse_BERTs/MRPC/Sparsity0.95/Epochs12$ contains the provided sparse BERT_base model for MRPC

TINYBERT_DIR=/SparseBERT/provided_sparse_BERTs/MRPC/Sparsity0.95/Epochs12
TASK_DIR=/SparseBERT/glue_data/MRPC
TASK_NAME=MRPC
OUTPUT_DIR=/SparseBERT/output_glue/tmp1

CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128

# Eval results: 
# acc = 0.8627450980392157
# acc_and_f1 = 0.8840752517223106
# eval_loss = 0.3557064888569025
# f1 = 0.9054054054054055
```


### Compress the pretrained BERT_base model and generate your generated sparse BERT_base model
```
# You can download the BERT_base_uncased model at <https://huggingface.co/bert-base-uncased>, which serves as the teacher model for knowledge distillation
# $/SparseBERT/pretrained_BERTs/BERT_base_uncased$ contains the pretrained BERT_base_uncased model, which is used as the initialization for the sparse BERT model
# Run 12 epochs
# $/SparseBERT/your_generated_sparse_BERTs/MRPC$ contains your generated sparse model


FT_BERT_BASE_DIR=/SparseBERT/finetuned_BERTs/bert_base_uncased_mrpc
GENERAL_TINYBERT_DIR=/SparseBERT/pretrained_BERTs/BERT_base_uncased
TASK_DIR=/SparseBERT/glue_data/MRPC
TASK_NAME=MRPC
TINYBERT_DIR=/SparseBERT/your_generated_sparse_BERTs/MRPC

CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill_prune_simultaneously.py --teacher_model $FT_BERT_BASE_DIR \
                       --student_model $GENERAL_TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $TINYBERT_DIR \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 12 \
                       --eval_step 200 \
                       --aug_train \
                       --do_lower_case \
                       --learning_rate 3e-5
```

### Get the results (eval set) of your generated sparse BERT_base model
```
TINYBERT_DIR=/SparseBERT/your_generated_sparse_BERTs/MRPC
TASK_DIR=/SparseBERT/glue_data/MRPC
TASK_NAME=MRPC
OUTPUT_DIR=/SparseBERT/output_glue/tmp1

CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128

# Eval results:
# acc = 0.8627450980392157
# acc_and_f1 = 0.8840752517223106
# eval_loss = 0.3557064888569025
# f1 = 0.9054054054054055
```

## QNLI

### Get the results (on eval set) of the finetuned BERT_base model
```
# You can download the finetuned BERT_base model at <https://huggingface.co/textattack/bert-base-uncased-QNLI>

TINYBERT_DIR=/SparseBERT/finetuned_BERTs/bert_base_uncased_qnli
TASK_DIR=/SparseBERT/glue_data/QNLI
TASK_NAME=QNLI
OUTPUT_DIR=/SparseBERT/output_glue/tmp1

CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128

# Eval results:
# acc = 0.9136005857587406
# eval_loss = 0.2559631625462694
```

### Get the results (eval set) of the provided sparse (x20) BERT_base model
```
TINYBERT_DIR=/SparseBERT/provided_sparse_BERTs/QNLI/Sparsity0.95/Epochs5
TASK_DIR=/SparseBERT/glue_data/QNLI
TASK_NAME=QNLI
OUTPUT_DIR=/SparseBERT/output_glue/tmp1

CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128


# Eval results
# acc = 0.9021
# eval_loss = 0.2798
```

## Check Sparsity

### Check the sparsity of the provided sparse (x20) BERT_base model (MRPC)
```
TINYBERT_DIR=/SparseBERT/provided_sparse_BERTs/MRPC/Sparsity0.95/Epochs12
TASK_DIR=/SparseBERT/glue_data/MRPC
TASK_NAME=MRPC
OUTPUT_DIR=/SparseBERT/output_glue/tmp1
CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill_calculate_sparsity.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128
```

### Check the sparsity of the provided sparse (x20) BERT_base model (QNLI)
```
TINYBERT_DIR=/SparseBERT/provided_sparse_BERTs/QNLI/Sparsity0.95/Epochs5
TASK_DIR=/SparseBERT/glue_data/QNLI
TASK_NAME=QNLI
OUTPUT_DIR=/SparseBERT/output_glue/tmp1
CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill_calculate_sparsity.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128
```

### Check the sparsity of the your generated sparse (x20) BERT_base model (MRPC)
```
TINYBERT_DIR=/SparseBERT/your_generated_sparse_BERTs/MRPC
TASK_DIR=/SparseBERT/glue_data/MRPC
TASK_NAME=MRPC
OUTPUT_DIR=/SparseBERT/output_glue/tmp1
CUDA_VISIBLE_DEVICES=0 python /SparseBERT/main_functions/task_distill_calculate_sparsity.py --do_eval \
                       --student_model $TINYBERT_DIR \
                       --data_dir $TASK_DIR \
                       --task_name $TASK_NAME \
                       --output_dir $OUTPUT_DIR \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128
```


## Acknowledgements
Our code is developed based on [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) and [Transformers](https://github.com/huggingface/transformers).