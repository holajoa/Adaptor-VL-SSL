export WANDB_DIR=/root/autodl-tmp/wandb/
export WANDB_DATA_DIR=/root/autodl-tmp/wandb/
export SAVED_MODEL_DIR="~/adaptor-thesis/trained_models/pretrain"
export VERSION=3
for VISION_MODEL in "resnet-ae" # "dinov2-b" "dinov2-s" 
do
    for TEXT_MODEL in  "bert" "biobert" "clinicalbert" "cxrbert" "pubmedbert"
    do
        python ./pretrain.py --n_gpus=4 --vision_model $VISION_MODEL --text_model $TEXT_MODEL --num_layers 2 --batch_size 1024 --data_pct 1.0 --num_workers 4 --num_train_epochs 50 --seed 42 --lr 2e-5 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_v${VERSION} --project_name adaptor_pretrain_v${VERSION} --wandb  | tee logs/pretrain_${VISION_MODEL}_${TEXT_MODEL}.txt
        wandb artifact cache cleanup 1MB
    done
done
/usr/bin/nvidia-smi
uptime
