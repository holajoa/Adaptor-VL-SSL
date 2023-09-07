export WANDB_DIR=/root/autodl-tmp/wandb/
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/
export SAVED_MODEL_DIR="/root/adaptor-thesis/trained_models/segment"
export DATASET="rsna"
export PROJECT_POSTFIX="_batchsize4"
for DATA_PCT in 1.0
do
    for TEXT_MODEL in "bert" "cxrbert" # "pubmedbert"  "biobert" "clinicalbert"  
    do
        for VISION_MODEL in "dinov2-s"  # "resnet-ae" "dinov2-b" 
        do
            echo ${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}
            python ./segment.py --n_gpus 4 --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 4 --data_pct $DATA_PCT --num_workers 1 --num_train_epochs 100 --seed 42 --lr 5e-4 --weight_decay 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --postfix v2 --pretrain_wandb_project_name adaptor_pretrain_2_layers --wandb --project_name adaptor_segment_2_layers${PROJECT_POSTFIX} --check_val_every_n_epochs 3 --patience_epochs 10 | tee logs/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime
