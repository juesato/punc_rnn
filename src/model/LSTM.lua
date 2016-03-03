local LSTM = {}
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout)
    dropout = dropout or 0 

    -- there will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1,n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}
    for L = 1,n do
        -- c,h from previos timesteps
        local prev_h = inputs[L*2+1]
        local prev_c = inputs[L*2]
        if L == 1 then
            -- To understand what's going on here with two calls, nngraph overrides __call__() function
            -- http://rnduja.github.io/2015/10/07/deep_learning_with_torch_step_4_nngraph/ 
            
            onehot = OneHot(input_size)(inputs[1]):annotate{name='onehot'}
            x = nn.Linear(input_size, 300)(onehot) 
            -- TODO: this should be LookupTable tbh.
            -- TODO: ignore any 0 inputs
            
            -- input_size_L = input_size
            -- x = inputs[1]
            -- x = nn.LookupTable(input_size, 300)(inputs[i]) -- embedding size 300
            input_size_L = 300
        else 
            x = outputs[(L-1)*2] 
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end

        -- evaluate the input sums at once for efficiency
        -- followed by some Reshape / SplitTable magic
        local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        -- perform the LSTM update, cell state
        local next_c           = nn.CAddTable()({ 
                nn.CMulTable()({forget_gate, prev_c}),
                nn.CMulTable()({in_gate,     in_transform})
            })
        -- gated cells form the output, hidden_state
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
        
        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

    -- set up the decoder
    local top_h = outputs[#outputs]
    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
    local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end

return LSTM

