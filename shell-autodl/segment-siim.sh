export WANDB_DIR=/root/autodl-tmp/wandb/
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/
export SAVED_MODEL_DIR="/root/adaptor-thesis/trained_models/segment"
export DATASET="siim"
export SEED=1117
export VERSION=3
for DATA_PCT in 0.1
do
    for TEXT_MODEL in "clinicalbert"  # "cxrbert" "pubmedbert"  "bert" "biobert" 
    do
        for VISION_MODEL in   "dinov2-b" #  "dinov2-s" "resnet-ae"  
        do  
            source /etc/network_turbo
            echo ${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}
            python ./segment.py --original_dice_loss --n_gpus 4 --seed $SEED --dataset $DATASET --crop_size 896 --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 4 --data_pct $DATA_PCT --num_workers 8 --num_train_epochs 100 --lr 5e-4 --weight_decay 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb --postfix v${VERSION} --pretrain_wandb_project_name adaptor_pretrain_v${VERSION} --wandb --project_name adaptor_segment_v${VERSION} --check_val_every_n_epochs 3 --patience_epochs 10 | tee logs/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
            wandb artifact cache cleanup 1GB
        done
    done
done