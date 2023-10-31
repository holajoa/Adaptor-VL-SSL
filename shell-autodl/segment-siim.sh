export WANDB_DIR=/root/autodl-tmp/wandb/
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/
export SAVED_MODEL_DIR="/root/adaptor-thesis/trained_models/segment"
export DATASET="siim"
export PROJECT_POSTFIX=""
export SEED=1024
export DATA_PCT=1.0
for TEXT_MODEL in "bert" "biobert" "clinicalbert" "cxrbert" "pubmedbert"
do
    for VISION_MODEL in "resnet-ae" "dinov2-b" "dinov2-s" 
    do
        echo ${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}
        python ./segment.py  --seed $SEED --dataset $DATASET --crop_size 896 --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 16 --data_pct $DATA_PCT --num_workers 1 --num_train_epochs 80 --lr 5e-4 --weight_decay 0.05 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb --postfix v2 --pretrain_wandb_project_name adaptor_pretrain_2_layers --wandb --project_name adaptor_segment_2_layers${PROJECT_POSTFIX} --check_val_every_n_epochs 5 --patience_epochs 20
        wandb artifact cache cleanup 1GB
    done
done
