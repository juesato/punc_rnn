require 'rnn'
require 'lfs'
require 'util.utils'

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


-- DEBUGGING FUNCTIONS --

local _ = require 'moses'
local updateOutputLinear = nn.Linear.updateOutput

-- function nn.Linear.updateOutput(self, input)
--     assert(not _.isNaN(input:sum()))
--     local output = updateOutputLinear(self, input)
--     assert(not _.isNaN(output:sum()))
--     return output
-- end

-- local updateOutputSequencer = nn.Sequencer.updateOutput
-- 
-- function nn.Sequencer.updateOutput(self, input)
--     assert(not _.isNaN(input:sum()))
--     local output = updateOutputSequencer(self, input)
--     assert(not _.isNaN(output:sum()))
--     return output
-- end
 
local updateOutputLookup = nn.LookupTableMaskZero.updateOutput
  
function nn.LookupTableMaskZero.updateOutput(self, input)
    assert(not _.isNaN(input:sum()))
    local output = updateOutputLookup(self, input)
    assert(not _.isNaN(output:sum()))
    return output
end

local updateOutputLSTM = nn.LSTM.updateOutput

function nn.LSTM.updateOutput(self, input)
    assert(not _.isNaN(input:sum()))
    local output = updateOutputLSTM(self, input)
    assert(not _.isNaN(output:sum()))
    return output
end

local updateOutputSoftmax = nn.LogSoftMax.updateOutput

function nn.LogSoftMax.updateOutput(self, input)
    assert(not _.isNaN(input:sum()))
    local output = updateOutputSoftmax(self, input)
    assert(not _.isNaN(output:sum()))
    return output
end

-- ERASE THIS BLOCK ABOVE THIS LINE LATER --




function prepro(x,y)
    local d1 = #x
    local x_size = x[1]:size(1)
    local y_size = y[1]:size(1)
    -- print("shape", d1, x_size)
    x = nn.JoinTable(1):forward(x)
    y = nn.JoinTable(1):forward(y)
    x = x:reshape(d1, x_size)
    y = y:reshape(d1, y_size)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing. 
    y = y:transpose(1,2):contiguous()
    x = x:double():cuda()
    y = y:double():cuda()

    inputs, outputs = {}, {}
    for i=1,x:size(1) do
    	inputs[i] = x[i]
    	outputs[i] = y[i]
    	local zeros = torch.eq(outputs[i], 0):double():cuda()
    	outputs[i]:add(1, zeros)
    end
    return inputs, outputs
end

--- CREATE RNN MODEL

local MinibatchLoader = require 'util.data_utils' -- MinibatchLoader
local loader = MinibatchLoader.create(opt.data_dir .. 'corpus.txt', config.batchSize, config.split_sizes)
local inputSize = loader.vocab_size + 1


loader:reset_batch_pointer(1, 1501)

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
	    -- print(torch.type(outputs[1]), torch.type(targets[1]))

	    local err = criterion:forward(outputs, targets)
	    local numZeros = 0

	    local totalTimeSteps = 0
	    -- compute precision/recall/f1
	    for j=1,#outputs do
		    local probs, preds = torch.max(outputs[j], 2)
		    -- local zeros = torch.ByteTensor(inputs[j]:size()):copy(torch.eq(inputs[j], 0)):nonzero()
		    -- for k=1,zeros:size(1) do
		    -- 	print (outputs[j][zeros[k][1]])
		    -- end
		    local unmasked = torch.ne(inputs[j], 0)
		    totalTimeSteps = totalTimeSteps + unmasked:sum()

		    local curTargets = targets[j][unmasked]
		    local curPreds   = preds:resizeAs(targets[j])[unmasked]
	    	for k=1,#loader.idx_to_punc do
	    		-- local is_target = torch.eq(targets[j], k)
	    		-- local is_pred = torch.eq(pred, k):resizeAs(is_target)
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

	return (loss / n)
end



-- Loss function
criterion = nn.ClassNLLCriterion() 
criterion = nn.MaskZeroCriterion(criterion, 1)
criterion = nn.SequencerCriterion(criterion)
criterion:cuda()

if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    rnn = checkpoint.rnn
    print ("Loaded from batch # ", checkpoint.i, "Val loss was", checkpoint.val_loss)
    eval_model(2, 50)
    -- -- make sure the vocabs are the same
    -- local vocab_compatible = true
    -- local checkpoint_vocab_size = 0
    -- for c,i in pairs(checkpoint.vocab) do
    --     if not (vocab[c] == i) then
    --         vocab_compatible = false
    --     end
    --     checkpoint_vocab_size = checkpoint_vocab_size + 1
    -- end
    -- if not (checkpoint_vocab_size == vocab_size) then
    --     vocab_compatible = false
    --     print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    -- end
    -- assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    
    -- overwrite model settings based on checkpoint to ensure compatibility
    -- print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    -- opt.rnn_size = checkpoint.opt.rnn_size
    -- opt.num_layers = checkpoint.opt.num_layers
    -- opt.model = checkpoint.opt.model
    -- do_random_init = false
else
	rnn = nn.Sequential()

	lookup = nn.LookupTableMaskZero(inputSize, config.embeddingSize)
	rnn:add(nn.Sequencer(lookup))
	-- if opt.dropout then
	-- 	rnn:add(nn.Dropout(opt.dropoutProb))
	-- end

	-- RNN
	for i=1,config.num_layers do
		local merge = nn.JoinTable(1,1)
		if i == 1 then
			local fwd = nn.LSTM(config.embeddingSize, config.hiddenSize):maskZero(1)
			-- rnn:add(nn.GRU(config.embeddingSize, config.hiddenSize):maskZero(1))
			-- rnn:add(nn.GRU(config.embeddingSize, config.hiddenSize))
			rnn:add(nn.BiSequencer(fwd, fwd:clone(), merge))
		else
			local fwd = nn.LSTM(config.hiddenSize * 2, config.hiddenSize):maskZero(1)
			-- rnn:add(nn.GRU(config.hiddenSize, config.hiddenSize):maskZero(1))
			-- rnn:add(nn.GRU(config.hiddenSize, config.hiddenSize))
			rnn:add(nn.BiSequencer(fwd, fwd:clone(), merge))
		end
		if opt.dropout then
		   rnn:add(nn.Dropout(opt.dropoutProb))
		   -- rnn:insert(nn.Dropout(opt.dropoutProb), 1)
		end
	end
	-- rnn:add(nn.Linear(config.hiddenSize, config.outputSize))
	rnn:add(nn.Sequencer(nn.MaskZero(nn.Linear(config.hiddenSize * 2, config.outputSize), 1)))
	rnn:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

	-- use Sequencer for better data handling
	-- rnn = nn.Sequencer(rnn)
	rnn:cuda()
	print("Model :")
	print(rnn)
end

-- eval_model(2, 5)

-- TRAINING LOOP
train_losses, val_losses = {}, {}
local iterations = opt.max_epochs * loader.ntrain

for k = 1,iterations do 
	-- 1. grab training point
    local x, y = loader:next_batch(config.train_split_id)
    local inputs, targets = prepro(x,y)
    -- inputs, targets should have dimension seq_len x batch_size 
    -- 2. forward pass
    local outputs = rnn:forward(inputs)

    local err, inGrad = criterion:forward(outputs, targets)  

    if err ~= err then
	    print('loss is NaN.  This usually indicates a bug.')
	    print('Exiting.')
	    break
	end

    -- 3. backwards pass
    rnn:zeroGradParameters()

    local gradOutputs = criterion:backward(outputs, targets)
    local gradInputs = rnn:backward(inputs, gradOutputs)

    -- 4. updates parameters   
    rnn:updateParameters(opt.learningRate)

    -- every now and then or on last iteration
    if k % opt.eval_val_every == 0 or k == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_model(2, 50) -- 2 = validation
        val_losses[k] = val_loss

        local epoch = k / loader.ntrain
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.5f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.rnn = rnn
        checkpoint.opt = opt
        -- checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = k
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.word_to_idx
        torch.save(savefile, checkpoint)
    end

    if k % opt.print_every == 0 then
	    print('Iter: ' .. k .. '   Length: ' .. #inputs .. '   Avg Err: ' .. err / #inputs) -- batchSize is already corrected for
        -- check weights and gradients
        -- local params, gradParams = rnn:getParameters()
        -- local maxWeight = torch.abs(params):max()
        -- local maxGrad = torch.abs(gradParams):max()
        -- print ("max weight and gradient", maxWeight, maxGrad)
    end
   
    if k % 10 == 0 then collectgarbage() end

end
