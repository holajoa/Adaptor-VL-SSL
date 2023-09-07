export WANDB_DIR=/root/autodl-tmp/wandb/
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/
export SAVED_MODEL_DIR="/root/adaptor-thesis/trained_models/clf"
export DATASET="covidx"
export VERSION=3
for DATA_PCT in 0.1 0.01
do
    for TEXT_MODEL in "bert" "biobert" "clinicalbert" "cxrbert" "pubmedbert" 
    do
        for VISION_MODEL in "resnet-ae" "dinov2-s" "dinov2-b" 
        do
            source /etc/network_turbo
            python ./finetune.py --n_gpus 4 --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 128 --data_pct $DATA_PCT --num_workers 8 --num_train_epochs 50 --seed 1024 --lr 5e-4 --weight_decay 0.05 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --postfix v${VERSION} --pretrain_wandb_project_name adaptor_pretrain_v${VERSION} --wandb --project_name adaptor_finetune_v${VERSION} --check_val_every_n_epochs 1 --patience_epochs 5  | tee logs/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
            wandb artifact cache cleanup 1GB
        done
    done
done

