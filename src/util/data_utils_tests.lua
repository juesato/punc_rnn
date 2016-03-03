MinibatchLoader = require 'data_utils'

local function test_parse_tokenized_line()
	print("wtf")
	x,y = MinibatchLoader.parse_tokenized_line("help me , what is going on ? do I do something here . . . how come ! ")
	print(x,y)
end

function test_Minibatch_create()
	local minibatchLoader = MinibatchLoader.create('../data/corpus.txt', 10, 10, {.8, .1, .1})
	return true
end

function test_Minibatch_next_batch()
	local minibatchLoader = MinibatchLoader.create('../data/corpus.txt', 10, 10, {.8, .1, .1})
	print("get next batch")
	local curx, cury = minibatchLoader:next_batch(1)
	print("XDIMs", curx:size())
	print("YDIMs", cury:size())
end

function test_ints_to_text()
	print("test_ints_to_text")
	local minibatchLoader = MinibatchLoader.create('../data/corpus.txt', 10, 10, {.8, .1, .1})
	local curx, cury = minibatchLoader:next_batch(1)
	print(curx, cury)
	print(minibatchLoader.vocab_size, "VOCABSIZE")
	print(minibatchLoader:ints_to_text(curx, cury))
end

test_parse_tokenized_line()
-- test_Minibatch_create()
-- test_Minibatch_next_batch()
test_ints_to_text()