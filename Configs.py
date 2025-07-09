class Config(object):
    def __init__(self):
        self.momentum = 0.99
        self.quene_K = 1024*64
        self.temperature = 0.2
        self.n_views = 2
        self.device = 'cuda'
        # model configs
        self.input_channels = 2
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 130 

        # training configs
        self.num_epoch = 40
        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 200

        self.Context_Cont = Context_Cont_configs()
        self.CP = CP()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.1
        self.jitter_ratio = 0.05
        self.max_seg = 9


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class CP(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 39
