import datetime


class NetConfig:

    def __init__(self):
        self.InputShape = 224
        self.InputDType = "float32"
        self.NumDenseBlocks = 3
        self.NumLayers = [6, 12, 32]
        self.NumOfClasses = 2
        self.NumberOfFilters = 16
        self.GrowthRate = 12
        self.Eta = 0.001
        self.batch_size = 50
        self.validation_split = 0.2
        self.epochs = 100
        self.weight_decay = 0.0001
        self.momentum_term = 0.9


class DensNet121(NetConfig):
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 24, 16]
        self.L = 121


class DensNet169:
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 32, 32]
        self.L = 169


class DensNet201:
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 48, 32]
        self.L = 169


class DensNet264:
    def __init__(self):
        super().__init__()
        self.NumDenseBlocks = 4
        self.NumLayers = [6, 12, 64, 48]
        self.L = 169
