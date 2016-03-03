-- this is a class factory. It creates a OneHot class, parameterized by outputSize
-- an instance of a OneHot class can be created by passing in another value. Idk how it knows how to init though.
local OneHot, parent = torch.class('OneHot', 'nn.Module')

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
end

function OneHot:updateOutput(input)
  -- I very well might have the wrong thing here.
  -- Input
  --   input: A 1D Tensor of indices, converted to one-hots
  self.output:resize(input:size(1), self.outputSize):zero()
  for i=1,input:size(1) do
    if input[i] > 0 then
      self.output[i][input[i]] = 1
    end
  end
  return self.output
end