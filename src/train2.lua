-- Multi-variate time-series example 

require 'rnn'
require 'lfs'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a multivariate time-series model using RNN')
cmd:option('-data_dir','../data/','data directory. Should contain the file corpus.txt with input data')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('--hiddenSize', 10, 'number of hidden units used at output of the recurrent layer')
-- cmd:option('--batch_size', 32, 'number of training samples per batch')
-- cmd:option('--nIterations', 1000, 'max number of training iterations')
cmd:option('--learningRate', 0.001, 'learning rate')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-eval_val_every',500,'every how many iterations should we evaluate on validation data?')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')


cmd:text()
local opt = cmd:parse(arg or {})

local config = require('config.lua')

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

torch.manualSeed(opt.seed)
-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

function prepro(x,y)
    local d1 = #x
    local x_size = x[1]:size(1)
    local y_size = y[1]:size(1)
    print("shape", d1, x_size)
    x = nn.JoinTable(1):forward(x)
    y = nn.JoinTable(1):forward(y)
    x = x:reshape(d1, x_size)
    y = y:reshape(d1, y_size)

    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing. 
    y = y:transpose(1,2):contiguous()
    -- print("prepro2", x:size(), y:size())

    -- x = x:float():cuda()
    -- y = y:float():cuda()

    inputs, outputs = {}, {}
    for i=1,x:size(1) do
    	inputs[i] = x[i]
    	outputs[i] = y[i]
    	local zeros = torch.eq(outputs[i], 0):double()
    	outputs[i]:add(4, zeros)
    end
    return inputs, outputs
end

--- CREATE RNN MODEL

local MinibatchLoader = require 'util.data_utils' -- MinibatchLoader
local loader = MinibatchLoader.create(opt.data_dir .. 'corpus.txt', config.batchSize, config.split_sizes)
local inputSize = loader.vocab_size + 1

rnn = nn.Sequential()

lookup = nn.LookupTableMaskZero(inputSize, config.embeddingSize)
rnn:add(lookup)
if opt.dropout then
   rnn:insert(nn.Dropout(opt.dropoutProb), 1)
end

-- RNN
for i=1,config.num_layers do
	if i == 1 then
		rnn:add(nn.GRU(config.embeddingSize, config.hiddenSize):maskZero(1))
	else
		rnn:add(nn.GRU(config.hiddenSize, config.hiddenSize):maskZero(1))
	end
	if opt.dropout then
	   rnn:insert(nn.Dropout(opt.dropoutProb), 1)
	end
end
rnn:add(nn.MaskZero(nn.Linear(config.hiddenSize, config.outputSize), 1))
rnn:add(nn.LogSoftMax())

-- use Sequencer for better data handling
rnn = nn.Sequencer(rnn)

criterion = nn.ClassNLLCriterion() 
criterion = nn.MaskZeroCriterion(criterion, 1)
criterion = nn.SequencerCriterion(criterion)

print("Model :")
print(rnn)

local timer = torch.Timer()
timer:stop()
-- TRAINING LOOP
local iterations = opt.max_epochs * loader.ntrain

for k = 1,iterations do 
	-- 1. grab training point
    local x, y = loader:next_batch(config.train_split_id)
    local inputs, targets = prepro(x,y)
    -- inputs, targets should have dimension seq_len x batch_size 
    -- 2. forward pass
	timer:reset()
	timer:resume()

    local outputs = rnn:forward(inputs)

    timer:stop()
    print("Forward step", timer:time()) -- about 5 real
    timer:reset()
    timer:resume()


    -- print ("output shape", #outputs, outputs[1]:size(), torch.type(outputs[1]))
    -- print ("target shape", #targets, targets[1]:size(), torch.type(targets[1]))
    -- for i=1,#outputs do
    -- 	outputs[i] = outputs[i]:float():cuda()
    -- end
    -- print ("output shape", #outputs, outputs[1]:size(), torch.type(outputs[1]))

    local err = criterion:forward(outputs, targets)   

    -- 3. backwards pass
    rnn:zeroGradParameters()

    local gradOutputs = criterion:backward(outputs, targets)
    local gradInputs = rnn:backward(inputs, gradOutputs)

    timer:stop()
    print("Backward step", timer:time()) -- 9s real
    timer:reset()

    -- 4. updates parameters   
    rnn:updateParameters(opt.learningRate)

    -- every now and then or on last iteration
    if k % opt.eval_val_every == 0 or k == iterations then
        -- evaluate loss on validation data
        -- local val_loss = eval_split(2) -- 2 = validation
        -- val_losses[i] = val_loss
        local epoch = k / loader.ntrain
        local savefile = string.format('%s/lm_%s_epoch%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.rnn = rnn
        checkpoint.opt = opt
        -- checkpoint.train_losses = train_losses
        -- checkpoint.val_loss = val_loss
        -- checkpoint.val_losses = val_losses
        checkpoint.i = i
        -- checkpoint.epoch = epoch
        checkpoint.vocab = loader.word_to_idx
        torch.save(savefile, checkpoint)
    end

    if k % opt.print_every == 0 then
	    print('Iter: ' .. k .. '   Length: ' .. #inputs .. '   Avg Err: ' .. err / #inputs)

    end
   
    if k % 10 == 0 then collectgarbage() end

end