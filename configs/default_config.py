import os
from types import SimpleNamespace
import platform



cfg = SimpleNamespace(**{})
cfg.fold = 0

if platform.system().lower() == 'windows':
    cfg.data_dir = r"D:\Study\ml\kaggle\archive\3d_training_pipeline\data\results\train/"
    cfg.data_json_dir = cfg.data_dir + f"dataset_3d_fold_{cfg.fold}.json"
elif platform.system().lower() == 'linux':
    cfg.data_dir = "./data/results/train/"
    cfg.data_json_dir = cfg.data_dir + f"Ldataset_3d_fold_{cfg.fold}.json"


cfg.test_df = cfg.data_dir + "sample_submission.csv"
cfg.output_dir = "./output/weights/"
cfg.log_dir = "./logdir"

cfg.batch_size = 32
cfg.val_batch_size = 64
cfg.img_size = (96, 96, 64)
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 0.0

# training
cfg.lr = 3e-4
cfg.min_lr = 1e-5
cfg.weight_decay = 0
cfg.epochs = 15
cfg.seed = -1
cfg.eval_epochs = 1
cfg.start_cal_metric_epoch = 1
cfg.warmup = 1

# resources
cfg.mixed_precision = True
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu
cfg.num_workers = 0
cfg.weights = None

basic_cfg = cfg
