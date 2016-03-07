local lfs = require( "lfs" )

require 'util.string_utils'
require 'util.utils'
require 'torch'
require 'nn'

punc_ids = {}

punc_ids['.'] = 2
punc_ids[','] = 3
punc_ids['!'] = 2
punc_ids['?'] = 4
-- punc_ids['PAD_'] = 5
-- 1 is always NOPUNC
-- idx_to_punc = {'', '.',',','?','PAD_'}
idx_to_punc = {'', '.',',','?'}

CORPUS_FILEPATH = '../data/corpus.txt'
VOCAB_FILEPATH  = '../data/vocab.t7'
VOCAB_SIZE      = 30000
X_TENSORFILE    = '../data/tensors_x.t7'
Y_TENSORFILE    = '../data/tensors_y.t7'

PAD_X_STR = '_PAD'
PAD_Y_STR = 'PAD_'
PAD_X_ID  = 0
PAD_Y_ID  = 0

ENG_SOURCES = {NYT_NYT=true,APW_ENG=true,CNN_HDL=true,
ABC_WNT=true,NBC_NNW=true,MNB_NBW=true,PRI_TWD=true, VOA_ENG=true}


-- Execution order:
-- python script runs first to create corpus file
-- MinibatchLoader.create(params)

local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

function MinibatchLoader.parse_tokenized_line(tokenized_list)
	-- takes output of NLTK word_tokenize and converts into lists of words (x) and ints (y)
	if tokenized_list == nil then return {},{} end
	local trimmed = trimString(tokenized_list)    	
	local words = trimmed:split(" ")
	local x = {}
	local y = {}
	local i = 0
	for j,word in ipairs(words) do
		if punc_ids[word] ~= nil and y[i] == nil and i > 0 then
			-- found punctuation symbol, haven't already assigned punc here
			y[i] = punc_ids[word]
		else 
			if containsAlphanumerics(word) then
				i = i+1
				x[i] = word
			end
		end
	end
	local y_tensor = torch.ones(#x)
	for i,val in pairs(y) do
		y_tensor[i] = val
	end
	return x,y_tensor
end

function MinibatchLoader.create_vocab_mapping(in_corpusfile, out_vocabfile, vocab_size)
	-- parse words one at a time, take top vocab_size words
	local word_counts = defaultdict(0)
	local j=0
    for line in io.lines(in_corpusfile) do
    	if line == nil then break end
    	local words = trimString(line):split(" ")
    	-- print(#words)
    	for i,word in ipairs(words) do
    		if containsAlphanumerics(word) then
    			word_counts[word] = word_counts[word] + 1
    		end
    	end
    	j=j+1
    	if j%1000==0 then print("Processed",j, "lines\n") end
    end
    local i = 1
    local idx_to_word = {}
    for k,v in spairs_by_value(word_counts) do
    	if i > vocab_size then break end
    	idx_to_word[i] = k
    	i = i+1
    end
    torch.save(out_vocabfile, idx_to_word)

    print(idx_to_word[1], idx_to_word[2], idx_to_word[3], idx_to_word[4])
end

function MinibatchLoader.load_vocab_file(vocab_file)
	local idx_to_word = torch.load(vocab_file)
	local word_to_idx = {}
	for i,word in ipairs(idx_to_word) do
		word_to_idx[word] = i
	end
	return idx_to_word, word_to_idx
end	

function MinibatchLoader:create_tensor_file(corpus_text_path)
	print("Call to MinibatchLoader:create_tensor_file") 

	local x_data = {}
	local y_data = {}
	local i = 1
	local tot_words = 0
    for line in io.lines(corpus_text_path) do
    	local x_words, y_tensor = MinibatchLoader.parse_tokenized_line(line)
    	if #x_words ~= 0 then -- ignore empty lines 
	    	local x_ints = self:text_to_ints(x_words)
	    	tot_words = tot_words + (#x_ints)[1]

	    	local sz = x_ints:size()[1]
	    	local j = 0
	    	-- print ("CHAR", j+self.max_len, "LOTTE", sz)
	    	while j + self.max_len < sz do
	    		x_data[i] = x_ints[{{j+1, j+self.max_len}}]
	    		y_data[i] = y_tensor[{{j+1, j+self.max_len}}]
	    		j = j + self.max_len
	    		i = i+1
	    	end
	    	-- print("size", sz)
    		x_data[i] = x_ints[{{j+1, sz}}]
    		y_data[i] = y_tensor[{{j+1, sz}}]
	    	i = i+1

	    	if i%1000==0 then print("Processed",i, "lines\nTotal words is", tot_words) end
    	end
    end

    print("About to save")
    torch.save(X_TENSORFILE, x_data)
    torch.save(Y_TENSORFILE, y_data)
end

function MinibatchLoader:load_batches(in_xtensorfile, in_ytensorfile, batch_size, split_sizes)
	local function pad_input(x, y, len)
		if x:size()[1] < len then
			local X = torch.cat(torch.Tensor(len - x:size()[1]):fill(PAD_X_ID), x)
			local Y = torch.cat(torch.Tensor(len - y:size()[1]):fill(PAD_Y_ID), y)
			return X,Y
		else
			return x,y
		end
	end

	local x_data = torch.load(in_xtensorfile) -- so these are now torch tensors apparently
	local y_data = torch.load(in_ytensorfile)

	print ("total size", #x_data)

	local x_buckets = defaultdict_from_fxn(function () return {} end)
	local y_buckets = defaultdict_from_fxn(function () return {} end)

	for i=1,#x_data do
		if x_data[i]:size(1) > 5 then -- ignore anything shorter than 5 words
			-- print ("numwords", x_data[i]:size(1))
			local x_len = x_data[i]:size()[1]
			for j=1,#self.bucket_lens do -- find the right bucket length
				if self.bucket_lens[j][1] < x_len and x_len <= self.bucket_lens[j][2] then
					local x_cur, y_cur = pad_input(x_data[i],y_data[i],self.bucket_lens[j][2])
					x_buckets[j][#(x_buckets[j]) + 1] = x_cur
					y_buckets[j][#(y_buckets[j]) + 1] = y_cur
				end
			end
			if x_data[i]:size(1) > self.bucket_lens[#self.bucket_lens][2] then
				print("WARNING: Input sequence with length", #x_data[i], "is too long. Did you cut inputs when generating " .. in_xtensorfile .. '?\n')
			end
		else
			-- print("Skipping short sequence", x_data[i])
		end
	end

	-- batching

	-- create one set of batches for so it's x_batches[split_index].
	-- then you can shuffle randomly so that the sizes are variable length.

	local x_batches = {}
	local y_batches = {}
	print ("Data points per bucket", #x_buckets[1], #x_buckets[2], #x_buckets[3], #x_buckets[4])

	for i=1,#x_buckets do -- iterate through each bucket length
		print ("Bucket", i, #(x_buckets[i]), torch.type(x_buckets[i][1]))
		local j = 0
		while j+batch_size < #x_buckets[i] do
			-- x_batches[i][#x_batches[i] + 1] = x_buckets[{{j+1, j+batch_size}}] 
			-- y_batches[i][#y_batches[i] + 1] = y_buckets[{{j+1, j+batch_size}}] 
			x_batches[#x_batches + 1] = table.slice(x_buckets[i], j+1, j+batch_size) 
			y_batches[#y_batches + 1] = table.slice(y_buckets[i], j+1, j+batch_size) 
			j = j+batch_size
		end
	end

	-- randomly permute the order, so that we have variable lengths everywhere
	local perm = torch.randperm(#x_batches)

	local x_shuffled = {}
	local y_shuffled = {}
	for i=1,perm:size(1) do
		local v = perm[i]
		x_shuffled[i] = x_batches[v]
		y_shuffled[i] = y_batches[v]
	end

	return x_shuffled, y_shuffled
end

function MinibatchLoader.create(corpus_text_path, batch_size, split_fractions)
	-- Constructor method
	-- Inputs:
	--   corpus_text_path: path to corpus file, formatted as tokenized text, with one data point per line. As outputted by Python script
	--   batch_size: an int, the batch size (unused)
	--   split_fractions: an array of 3 floats, e.g. {.8, .1, .1} for train, val, test

    local self = {}
    setmetatable(self, MinibatchLoader)

    self.batch_size  = batch_size
    self.bucket_lens = {{0,25},{25,50},{50,75},{75,100}}
    self.max_len     = self.bucket_lens[#self.bucket_lens][2]

    -- Creating vocabulary mapping
	if not file_exists(VOCAB_FILEPATH) then
		print("Creating file vocab.t7")
		MinibatchLoader.create_vocab_mapping(corpus_text_path, VOCAB_FILEPATH, VOCAB_SIZE)
	end
	self.idx_to_word, self.word_to_idx = MinibatchLoader.load_vocab_file(VOCAB_FILEPATH)
	-- self.vocab_mapping = self.word_to_idx
	self.vocab_size    = #self.idx_to_word
	self.OOV_idx       = self.vocab_size + 1

	self.punc_to_idx = punc_ids
	self.idx_to_punc = idx_to_punc

	-- Creating tensors and data points
	if not file_exists(X_TENSORFILE) or not file_exists(Y_TENSORFILE) then
		print("Creating tensors_x.t7 and tensors_y.t7")
		self:create_tensor_file(corpus_text_path)
	end

	print("Loading data from", X_TENSORFILE)

	self.x_batches, self.y_batches = self:load_batches(X_TENSORFILE,Y_TENSORFILE, batch_size, split_fractions)
	-- self.x_batches   = torch.load(X_TENSORFILE) -- lua array of ints
	-- self.y_batches   = torch.load(Y_TENSORFILE) -- lua array of ints

	print("Finished loading data")

	self.bucket_lens = {}
	for i=1,#self.x_batches do
		self.bucket_lens[i] = #self.x_batches[i]
	end
	self.data_points = sum(self.bucket_lens)

	self.nbatches    = #self.x_batches
    self.ntrain      = math.floor(self.nbatches * split_fractions[1])
    self.nval        = math.floor(self.nbatches * split_fractions[2])
    self.ntest       = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_idx   = {0,0,0}
    -- self.batch_idx   = {{0,0,0,0},{0,0,0,0},{0,0,0,0}}
    return self
end

function MinibatchLoader:text_to_ints(x_textin)
	-- take output of parse_token_list() and generate ints
	local x = torch.Tensor(#x_textin):zero()
	for i,word in ipairs(x_textin) do
		x[i] = self.word_to_idx[word] or self.OOV_idx
	end
	return x
end

function MinibatchLoader:ints_to_text(x_ints, y_ints)
	-- y_tensors can be all 0s to have no punctuation
	-- if debug, at prediction - output TEXT of input, TEXT of predicted output
	-- print("MinibatchLoader:ints_to_text")
	local out = ''
	for i=1,#x_ints do
		out = out .. (self.idx_to_word[x_ints[i]] or 'OOV') .. ' '
		-- print("Y", y_ints[i])
		-- out = out .. self.idx_to_punc[1] .. ' '
		out = out .. self.idx_to_punc[y_ints[i]] .. ' '
	end
	return out
end

function MinibatchLoader:reset_batch_pointer(split_index, batch_index)
    local batch_index = batch_index or 0
    self.batch_idx[split_index] = batch_index
end

-- function MinibatchLoader:next_batch(split_index)
--     if self.split_sizes[split_index] == 0 then
--         -- perform a check here to make sure the user isn't screwing something up
--         local split_names = {'train', 'val', 'test'}
--         print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
--         os.exit() -- crash violently
--     end
--     -- split_index is integer: 1 = train, 2 = val, 3 = test
--     self.batch_idx[split_index] = self.batch_idx[split_index] + 1
--     if self.batch_idx[split_index] > self.split_sizes[split_index] then
--         self.batch_idx[split_index] = 1 -- cycle around to beginning
--     end
--     -- pull out the correct next batch
--     local idx = self.batch_idx[split_index]
--     if split_index == 2 then idx = idx + self.ntrain end -- offset by train set size
--     if split_index == 3 then idx = idx + self.ntrain + self.nval end -- offset by train + val

--     -- print("X data", self.x_batches[idx])
--     local vect_sz = self.vocab_size + 1
--     return self.x_batches[idx], self.y_batches[idx]
-- end

function MinibatchLoader:next_batch(split_index)

    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_index] = self.batch_idx[split_index] + 1
    if self.batch_idx[split_index] > self.split_sizes[split_index] then
        self.batch_idx[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_index]
    if split_index == 2 then idx = idx + self.ntrain end -- offset by train set size
    if split_index == 3 then idx = idx + self.ntrain + self.nval end -- offset by train + val

    -- print("X data", self.x_batches[idx])
    local vect_sz = self.vocab_size + 1
    return self.x_batches[idx], self.y_batches[idx]
end

return MinibatchLoader