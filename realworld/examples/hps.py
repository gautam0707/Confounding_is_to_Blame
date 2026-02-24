cfgs_best_hps = {
    ('mlp', 'diabetes_readmission'):
    {
        'num_layers':       2,
        'd_hidden':         256,
        'dropouts':         0.488,
        'lr':               0.0034,
        'weight_decay':     0.0282,
        'batch_size':       4096,
        'n_epochs':        10
    },

 ('mlp', 'acsfoodstamps'):
    {
        'num_layers':       7,
        'd_hidden':         256,
        'dropouts':         0.1487,
        'lr':               0.0063,
        'weight_decay':     0.1297,
        'batch_size':       4096,
        'n_epochs':        100
    },
 ('mlp','brfss_diabetes'):{
        'num_layers':       4,
        'd_hidden':         128,
        'dropouts':         0.0833,
        'lr':               0.0005,
        'weight_decay':     0.0001,
        'batch_size':       4096,
        'n_epochs':        40
    },
 ('mlp','acsincome'):{
        'num_layers':       2,
        'd_hidden':         64,
        'dropouts':         0.3103,
        'lr':               0.0001,
        'weight_decay':     0.0000,
        'batch_size':       4096,
        'n_epochs':        20
    },
     ('mlp','acsunemployment'):{
        'num_layers':       4,
        'd_hidden':         512,
        'dropouts':         0.0144,
        'lr':               0.0015,
        'weight_decay':     0.3650,
        'batch_size':       4096,
        'n_epochs':        10
    },
    ('mlp','assistments'):{
         'num_layers':       5,
        'd_hidden':         128,
        'dropouts':         0.0562,
        'lr':               0.0004,
        'weight_decay':     0.0046,
        'batch_size':       4096,
        'n_epochs':        80
    },

    ('group_dro','diabetes_readmission'):{
        'num_layers':               3,
        'd_hidden':                 512,
        'group_weights_step_size':  0.0342,
        'dropouts':                 0.1083,
        'lr':                       0.0115,
        'weight_decay':             0.0282,
        'batch_size':               4096,
        'n_epochs':                 40
    },
   
('group_dro','acsfoodstamps'):{
        'num_layers':               4,
        'd_hidden':                 1024,
        'group_weights_step_size':  0.1343,
        'dropouts':                 0.1342,
        'lr':                       0.0046,
        'weight_decay':             0.1292,
        'batch_size':               4096,
        'n_epochs':                 30
},

    ('group_dro','brfss_diabetes'):{
        'num_layers':               4,
        'd_hidden':                 512,
        'group_weights_step_size':  0.0008,
        'dropouts':                 0.1216,
        'lr':                       0.0083,
        'weight_decay':             0.0001,
        'batch_size':               4096,
        'n_epochs':                 90
    },
   
    ('group_dro','acsincome'):{
        'num_layers':               4,
        'd_hidden':                 512,
        'group_weights_step_size':  0.0008,
        'dropouts':                 0.1216,
        'lr':                       0.0083,
        'weight_decay':             0.0001,
        'batch_size':               4096,
        'n_epochs':                 90
    },   
    ('group_dro','acsunemployment'):{
        'num_layers':               1,
        'd_hidden':                 128,
        'group_weights_step_size':  0.0446,
        'dropouts':                 0.1499,
        'lr':                       0.0006,
        'weight_decay':             0.0001,
        'batch_size':               4096,
        'n_epochs':                 10
    },
    ('group_dro','assistments'):{
        'num_layers':               6,
        'd_hidden':                 1024,
        'group_weights_step_size':  0.0009,
        'dropouts':                 0.4239,
        'lr':                       0.0004,
        'weight_decay':             0.0001,
        'batch_size':               4096,
        'n_epochs':                 90
    },
    ('irm','diabetes_readmission'): {
        'num_layers':               4,
        'd_hidden':                 256,
        'dropouts':                 0.2230,
        'irm_lambda':               2.8819,
        'irm_penalty_anneal_iters': 7.5010,
        'lr':                       0.0083,
        'weight_decay':             0.0035,
        'batch_size':               4096,
        'n_epochs':                 90
    },
    ('irm','acsfoodstamps'): {
        'num_layers':               4,
        'd_hidden':                 256,
        'dropouts':                 0.2230,
        'irm_lambda':               2.8819,
        'irm_penalty_anneal_iters': 7.5010,
        'lr':                       0.0083,
        'weight_decay':             0.0035,
        'batch_size':               4096,
        'n_epochs':                 90
    },
    ('irm','brfss_diabetes'):{
        'num_layers':               4,
        'd_hidden':                 256,
        'dropouts':                 0.4839,
        'irm_lambda':               1822.90,
        'irm_penalty_anneal_iters': 8.4789,
        'lr':                       0.0762,
        'weight_decay':             0.0000,
        'batch_size':               4096,
        'n_epochs':                 70
},
('irm','acsincome'): {
        'num_layers':               4,
        'd_hidden':                 256,
        'dropouts':                 0.2230,
        'irm_lambda':               2.8819,
        'irm_penalty_anneal_iters': 7.5010,
        'lr':                       0.0083,
        'weight_decay':             0.0035,
        'batch_size':               4096,
        'n_epochs':                 90
    },
    ('irm','acsunemployment'): {
        'num_layers':               4,
        'd_hidden':                 256,
        'dropouts':                 0.2230,
        'irm_lambda':               2.8819,
        'irm_penalty_anneal_iters': 7.5010,
        'lr':                       0.0083,
        'weight_decay':             0.0035,
        'batch_size':               4096,
        'n_epochs':                 90
    },
    ('vrex','diabetes_readmission'): {
        'num_layers':               6,
        'd_hidden':                 256,
        'vrex_penalty_anneal_iters':1.5620,
        'vrex_lambda':              5.7354,
        'dropouts':                 0.1874,
        'lr':                       0.0074,
        'weight_decay':             0.8254,
        'batch_size':               4096,
        'n_epochs':                 70
    },
    ('vrex','acsfoodstamps'): {
        'num_layers':               3,
        'd_hidden':                 128,
        'vrex_penalty_anneal_iters':104.34893,
        'vrex_lambda':              9.361638,
        'dropouts':                 0.26036,
        'lr':                       0.00005,
        'weight_decay':             0.001488,
        'batch_size':               4096,
        'n_epochs':                 80
    },
    ('vrex','brfss_diabetes'): {
        'num_layers':               7,
        'd_hidden':                 512,
        'vrex_penalty_anneal_iters':4895.2029,
        'vrex_lambda':              331.344,
        'dropouts':                 0.2815,
        'lr':                       0.0038,
        'weight_decay':             0.0000,
        'batch_size':               4096,
        'n_epochs':                 100
    },
    ('vrex','acsincome'):{
        'num_layers':               6,
        'd_hidden':                 256,
        'vrex_penalty_anneal_iters':8580.826,
        'vrex_lambda':              2685.1824,
        'dropouts':                 0.1266,
        'lr':                       0.0006,
        'weight_decay':             0.0008,
        'batch_size':               4096,
        'n_epochs':                 45

    },
    ('vrex','acsunemployment'):{
        'num_layers':               3,
        'd_hidden':                 512,
        'vrex_penalty_anneal_iters':99.1488,
        'vrex_lambda':              15.6214,
        'dropouts':                 0.14485,
        'lr':                       0.0040,
        'weight_decay':             0.09932,
        'batch_size':               4096,
        'n_epochs':                 65

    }
    }

# cfgs = {
#     'irm': {
#         'num_layers':               [2, 3, 4],
#         'd_hidden':                 [256, 512, 1024],
#         'dropouts':                 [0.0, 0.1, 0.2],
#         'irm_lambda':               [0.01, 0.1, 0.05],
#         'irm_penalty_anneal_iters':[1, 2,3],
#         'lr':                       [0.01, 0.02, 0.05],
#         'weight_decay':             [0.01, 0.001, 0.0001],
#         'batch_size':               [4096],
#         'n_epochs':                 [1,2,3]
#     },

#     'vrex': {
#         'num_layers':               [2, 3, 4],
#         'd_hidden':                 [256, 512, 1024],
#         'vrex_penalty_anneal_iters':[1,2,3],
#         'vrex_lambda':              [0.1, 10, 100],
#         'dropouts':                 [0.0, 0.1, 0.2],
#         'lr':                       [0.01, 0.02, 0.05],
#         'weight_decay':             [0.01, 0.001, 0.0001],
#         'batch_size':               [4096],
#         'n_epochs':                 [1,2,3]
#     },

#     'xgb': {
#         'learning_rate':       [0.3, 0.2, 0.1],
#         'max_depth':           [6, 4, 5],
#         'min_child_weight':    [1,10,0.1],
#         'gamma':               [0.01, 0.001, 0.0001],
#         'subsample':           [0.5,0.6,0.7],
#         'colsample_bytree':    [0.5,0.6,0.7],
#         'reg_alpha':           [0.01, 0.001, 0.0001],
#         'reg_lambda':          [0.01, 0.001, 0.0001]
#     }
# }
