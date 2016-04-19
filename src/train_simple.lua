require 'rnn'
require 'lfs'
require 'util.utils'
require 'dpnn'
require 'optim'
require 'cunn'

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
cmd:option('-seed',125,'torch manual random number generator seed')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-eval_val_every',500,'every how many iterations should we evaluate on validation data?')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')


cmd:text()
local opt = cmd:parse(arg or {})

local config = require('config.lua')

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

if opt.seed then torch.manualSeed(opt.seed) end
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
    local x = nn.JoinTable(1):forward(x)
    local y = nn.JoinTable(1):forward(y)
    local x = x:reshape(d1, x_size)
    local y = y:reshape(d1, y_size)
    local x = x:transpose(1,2):contiguous()
    local y = y:transpose(1,2):contiguous()
    local x = x:double():cuda()
    local y = y:double():cuda()

    local inputs, outputs = {}, {}
    for i=1,x:size(1) do
        inputs[i] = x[i]
        outputs[i] = torch.gt(inputs[i], 300):add(1)
        --inputs[i] = torch.Tensor(x[i]:size())
        --inputs[i]:random(1,4)
        --inputs[i]:double():cuda()
        --outputs[i] = inputs[i]:clone()
        inputs[i] = inputs[i]:cuda()
        outputs[i] = outputs[i]:cuda()

    	--local zeros = torch.eq(outputs[i], 0):double():cuda()
    	--outputs[i]:add(1, zeros) -- these should be masked out anyways, but the outputs need to be nonzero to avoid an error with the NLLClassCriterion
        -- outputs[i] = torch.ge(inputs[i], 100):add(1)
    end
    --print ("JCHOW", inputs[1]:size(), inputs[1])
    --os.exit(1)
    return inputs, outputs
end

--- CREATE RNN MODEL

local MinibatchLoader = require 'util.data_utils' -- MinibatchLoader
local loader = MinibatchLoader.create(opt.data_dir .. 'corpus.txt', config.batchSize, config.split_sizes)
local inputSize = loader.vocab_size + 1

function eval_model(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)	
    local n = loader.split_sizes[split_idx]
    if max_batches ~= nil then n = math.min(max_batches, n) end	
	loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
	rnn:evaluate() -- put in eval mode so that dropout works properly

	true_pos, false_pos, false_neg = {}, {}, {}
	for i=1,#loader.idx_to_punc do
		true_pos[i] = 0
		false_pos[i] = 0
		false_neg[i] = 0
	end

    for i=1,n do
	    local x, y = loader:next_batch(split_idx)
	    local inputs, targets = prepro(x,y)

	    local outputs = rnn:forward(inputs)
	    local err = criterion:forward(outputs, targets)
	    local numZeros = 0

	    local totalTimeSteps = 0
	    -- compute precision/recall/f1
	    for j=1,#outputs do
		    local probs, preds = torch.max(outputs[j], 2)
		    local unmasked = torch.ne(inputs[j], 0)
		    totalTimeSteps = totalTimeSteps + unmasked:sum()

		    local curTargets = targets[j][unmasked]
		    local curPreds   = preds:resizeAs(targets[j])[unmasked]
	    	for k=1,#loader.idx_to_punc do
	    		local is_target = torch.eq(curTargets, k)
	    		local is_pred = torch.eq(curPreds, k)
	    		local true_posits = torch.dot(is_target, is_pred)
	    		local false_posits = is_pred:sum() - true_posits
	    		local false_negs = is_target:sum() - true_posits
	    		true_pos[k] = true_pos[k] + true_posits
	    		false_pos[k] = false_pos[k] + false_posits
	    		false_neg[k] = false_neg[k] + false_negs
	    	end
		end
	    loss = loss + err * config.batchSize / (#outputs * config.batchSize - numZeros)
	end
	print("Avg loss ", loss / n)
	for i=1,#loader.idx_to_punc do
		print("For punctuation mark", loader.idx_to_punc[i])
		local prec = true_pos[i] / (true_pos[i] + false_pos[i])
		local recall = true_pos[i] / (true_pos[i] + false_neg[i])
		local f1 = 2 * prec * recall / (prec + recall)
		print ("Precision", prec)
		print ("Recall", recall)
		print ("F1 Score", f1)
	end
    rnn:training()

	return (loss / n)
end

-- Loss function
criterion = nn.ClassNLLCriterion() 
criterion = nn.MaskZeroCriterion(criterion, 1)
criterion = nn.SequencerCriterion(criterion)
criterion:cuda()

local start_iter = 10

if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    rnn = checkpoint.rnn
    start_iter = checkpoint.i + 1
    -- loader:reset_batch_pointer(1, checkpoint.i)
    print ("Loaded from batch # ", checkpoint.i, "Val loss was", checkpoint.val_loss)
else
	rnn = nn.Sequential()
    --require 'model/OneHot.lua'
    --rnn:add( nn.OneHot(inputSize) )
    lookup = nn.LookupTableMaskZero(inputSize, config.embeddingSize)
	--lookup = nn.LookupTableMaskZero(inputSize, config.outputSize)
    for i=1,inputSize-1 do
        lookup.weight[i] = loader.word_to_vec[loader.idx_to_word[i]]
    end 
	rnn:add(lookup)
	local fwd = nn.FastLSTM(config.embeddingSize, config.hiddenSize)
	rnn:add(fwd)
	rnn:add(nn.Linear(config.hiddenSize, config.outputSize))
    --rnn:add(nn.Linear(config.embeddingSize, config.outputSize))

	rnn:add(nn.LogSoftMax())
	rnn = nn.Sequencer(rnn)
	rnn:cuda()
	print("Model :")
	print(rnn)
end

-- TRAINING LOOP
train_losses, val_losses = {}, {}
local max_iterations = opt.max_epochs * loader.ntrain
loader:reset_batch_pointer(1, start_iter)

params, gradParams = rnn:getParameters()

function feval (params_new)
    if params ~= params_new then
        params:copy(params_new)
    end
    local x, y = loader:next_batch(config.train_split_id)
    local inputs, targets = prepro(x, y)
    batch_seq_len_glob = #inputs
    --print ("INPUTS[1]",torch.any(torch.eq(inputs[1], 0)), inputs[1])
    local outputs = rnn:forward(inputs)
    for i=1,#targets do targets[i]:cuda() end
    err, inGrad = criterion:forward(outputs, targets)
    print (err )
    if err ~= err then
        print('loss is NaN.  This usually indicates a bug.')
    end
    
    rnn:zeroGradParameters()
    local gradOutputs = criterion:backward(outputs, targets)
    local gradInputs = rnn:backward(inputs, gradOutputs)
    if gradParams:norm() > 10 then
        gradParams:div(gradParams:norm() / 10)
    end
    --print ("GRADPARAMS", gradParams[{{1,12}}])
    return err, gradParams
end    

adam_params = {
    learningRate = .01
}
hyperparams = {
    learningRate = .1
}

for k=start_iter, max_iterations do
    --print ("ADAM", adam_params)
    _, fs = optim.adam(feval, params, adam_params)
    --local m = adam_params.m
    --print ("M", m[{{1,12}}])

        local maxWeight = torch.abs(params):max()
        local maxGrad = torch.abs(gradParams):max()
        local norm = gradParams:norm()
        print ("max weight and gradient", maxWeight, maxGrad, norm, hyperparams)

    if (k > 0 and k % opt.eval_val_every == 0) or k == max_iterations then
        print ("WEIGHTS", lookup.weight[{{1,6}}], lookup.weight[{{301, 306}}])
        local val_loss = 99
        local val_loss = eval_model(2, 50) -- 2 = validation
        --val_losses[k] = val_loss
        local epoch = k / loader.ntrain
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.5f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.rnn = rnn
        checkpoint.opt = opt
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = k
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.word_to_idx
        torch.save(savefile, checkpoint)
    end

    if k % opt.print_every == 0 then
        last_time = last_time or 0
        elapsed_time = os.clock() - last_time
        last_time = os.clock()
        print('Iter: ' .. k .. '   Length: ' .. batch_seq_len_glob .. '   Avg Err: ' .. err / batch_seq_len_glob .. "  Time: " .. elapsed_time)  -- batchSize is already corrected for
        --local updates = torch.cdiv(gradParams, torch.add(hyperparams.paramStd, 1e-12) )
        --print ("max Update", torch.abs(updates):max())
    end

    if k % 10 == 0 then collectgarbage() end
end
