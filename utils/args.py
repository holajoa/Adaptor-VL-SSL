import argparse


def get_train_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vision_model', type=str, help='Choose from [resnet-ae, swin-base]')
    parser.add_argument('--text_model', type=str, 
                        help='Choose from [bert, biobert, pubmedbert, cxrbert, clinicalbert]')
    parser.add_argument('--num_of_samples', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--force_rebuild_dataset', action='store_true', help='Whether to force rebuild dataset, if not can load pickled file if available')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_pct', type=float, default=1.0, help='percentage of data to use')
    parser.add_argument('--crop_size', type=int, default=224)

    parser.add_argument('--num_layers', type=int, default=1, help='number of transformer layers to use in adaptor')
    parser.add_argument('--projection_dim', type=int, default=768, help='dimension of projection head')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=200)

    parser.add_argument('--output_dir', type=str, default='./results', help='path to save model')
    
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--n_gpus', type=int, default=2, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=1117)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--cache_dir', default='/vol/bitbucket/jq619/.cache/huggingface/')
    return parser