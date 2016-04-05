require 'rnn'

branch1=nn.Sequential() :add(nn.Narrow(1,1,1)) :add(nn.LookupTable(2,5))
branch2=nn.Sequential() :add(nn.Narrow(1,2,1)) :add(nn.LookupTable(2,5))
mlp = nn.Concat(1) :add(branch1) :add(branch2)

t1 = torch.Tensor({1,2})
print(mlp:forward(t1))

mlp_seq = nn.Sequencer(mlp)
t2 = torch.Tensor({2,1})
print(mlp_seq:forward({t1,t2}))