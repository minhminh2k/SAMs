from box import Box

config = {
    "num_devices": 1, # 4
    "batch_size": 1, # 12
    "num_workers": 4, # 4
    "num_epochs": 30, # 20
    "eval_interval": 1,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4, 
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "out/training/epoch-017-f1_0.74-ckpt.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "data/images/train2017",
            "annotation_file": "data/annotations/instances_train2017.json"
        },
        "val": {
            "root_dir": "data/images/val2017",
            "annotation_file": "data/annotations/instances_val2017.json"
        }
    }
}

cfg = Box(config)