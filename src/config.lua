
local config = {}
config.__index = config

config.batchSize = 64
config.outputSize = 4
config.num_layers = 2
config.embeddingSize = 300
config.hiddenSize = 256
config.train_split_id = 1
config.valid_split_id = 2
config.test_split_id = 3
config.split_sizes = {0.8, 0.1, 0.1}


return config