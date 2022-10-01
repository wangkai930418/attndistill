DATASET='ImageNet'
TEACHER_MODEL_NAME=vit_small_patch8
STUDENT_MODEL_NAME=vit_small_patch16_teacher_small
WEIGHT_CLS=1.0
WEIGHT_ATTN=0.1
PORT=29501
PORJ_LAYER=4
PREFIX=dino
WARMUP=40
BATCH_SIZE=128
EPOCHS=801
ACCUM=1
BLR=0.00015
ATTN_T=10.0
SUBDIR=${PREFIX}_${TEACHER_MODEL_NAME}_to_${STUDENT_MODEL_NAME}_cls_${WEIGHT_CLS}_attn_${WEIGHT_ATTN}
OUTPUT_DIR=output_dir/${DATASET}/${SUBDIR}
# change the following two paths: dataset path and the teacher model saved path
IMAGENET_DIR=/data/datasets/${DATASET}/
TEACHER_RESUME=/data/datasets/SS_ViT/dino_vit/dino_deitsmall8_pretrain.pth
# STUDENT_RESUME=output_dir/${DATASET}/${SUBDIR}/checkpoint-300.pth

# modify the gpu setup to your case
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=2 dino_transfer.py \
    --batch_size ${BATCH_SIZE} \
    --teacher_model ${TEACHER_MODEL_NAME} \
    --student_model ${STUDENT_MODEL_NAME} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP} \
    --blr ${BLR} --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --save_frequency 100 \
    --accum_iter ${ACCUM} \
    --teacher_resume ${TEACHER_RESUME} \
    --weight_cls ${WEIGHT_CLS} \
    --weight_attn ${WEIGHT_ATTN} \
    --proj \
    --proj_layers ${PORJ_LAYER} \
    --dist_type cos \
    --port ${PORT} \
    --interpolate \
    # --student_resume ${STUDENT_RESUME} \
    # --aggregation \
    # --attn_T ${ATTN_T} \
    # --lmdb \
