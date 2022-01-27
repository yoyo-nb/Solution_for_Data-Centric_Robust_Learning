args_wideresnet = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.12,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 48,
}
args_preactresnet18 = {
    'epochs': 200,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.012,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 48,
}