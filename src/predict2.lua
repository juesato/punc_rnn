
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'rnn'

require 'cunn' -- necessary to load model?

require 'util.string_utils'
-- require 'util.misc'

local MinibatchLoader = require 'util.data_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
-- cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

opt = cmd:parse(arg)
local config = require('config.lua')

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

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

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
rnn = checkpoint.rnn
rnn:evaluate() -- put in eval mode so that dropout works properly

local input
local inp_as_toks
local BATCH_SIZE = 1
local DATA_DIR = '../data/'
local SEQ_LEN = 50
local loader = MinibatchLoader.create(DATA_DIR .. 'corpus.txt', 1, config.split_sizes)

repeat
    -- TODO: INIT HIDDEN STATE TO 0
    io.flush()
    io.write("Enter a sentence to add punctuation or 'exit' to exit:\n\n")
    io.flush()
    input=io.read()
    inp_as_toks=input:trim():split(' ')
    x = loader:text_to_ints(inp_as_toks)
    x = x:resize(x:size(1), 1)
    x = nn.SplitTable(1, 2):forward(x)
    y = rnn:forward(x)
    preds = {}
    for i=1,#y do
        print(y[i])
        prob, preds[i] = y[i]:max(2)
        preds[i] = preds[i][1][1]
        x[i] = x[i][1]
    end
    io.write(loader:ints_to_text(x, preds) .. '\n\n')
    io.flush()
until input=="exit"