require 'cutorch'
require 'cunn'
require 'rnn'

criterion = nn.ClassNLLCriterion() 
criterion = nn.MaskZeroCriterion(criterion, 1)
criterion = nn.SequencerCriterion(criterion):cuda()

outputs = {}
targets = {}
seq_length = 10
batch_size = 5
output_size = 3
for i=1,seq_length do		
	outputs[i] = torch.rand(batch_size, output_size):float():cuda()
	if i < 3 then
		outputs[i]:fill(0)
	end
	targets[i] = torch.Tensor(batch_size):fill(1):float():cuda()
	-- outputs[i] = torch.rand(batch_size, output_size)
	-- targets[i] = torch.Tensor(batch_size):fill(1)
end

local err = criterion:forward(outputs, targets)   
print(outputs[1])
print(criterion.criterion)
print(criterion:getStepCriterion(1))
print(criterion:getStepCriterion(3).zeroMask)

print (criterion.criterion.zeroMask)
print(err)