export WANDB_DIR=/root/autodl-tmp/wandb/
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/
export SAVED_MODEL_DIR=/root/adaptor-thesis/trained_models/segment_sweep
export DATASET="siim"
export CROP_SIZE=896
export MAX_EPOCHS=100
export DATA_PCT=1.0
export VISION_MODEL="dinov2-b"
export TEXT_MODEL="clinicalbert"
for BATCH_SIZE in 4
do
    for LR in 5e-4 2.5e-4
    do
        for WD in 1e-4 0.05 
        do
            python ./segment.py --n_gpus 4 --dataset $DATASET --crop_size $CROP_SIZE --vision_model $VISION_MODEL --text_model $TEXT_MODEL --data_pct $DATA_PCT --num_workers 8 --num_train_epochs $MAX_EPOCHS --seed 42 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --project_name "segment-sweep_${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}" --batch_size $BATCH_SIZE --lr $LR --weight_decay $WD --postfix v2 --pretrain_wandb_project_name adaptor_pretrain_2_layers --wandb --check_val_every_n_epochs 3 --patience_epochs 10 | tee logs/segment-sweep_${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
            wandb artifact cache cleanup 1GB
        done
    done
done
