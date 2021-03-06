function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function sequenceToTensor(s)
    for i=1,#s do
        local sz = torch.totable(s[i]:size())
        table.insert(sz,1)
        s[i]:view(unpack(sz))
    end
    return nn.JoinTable(1):forward(s)
end

function debugNansSequence(s)
    return debugNansTensor(sequenceToTensor(s))
end

function debugNansTensor(t)
    local nans = t:ne(t)
    if nans:sum() == 0 then
        return false
    else
        print ("Shape of Tensor is", t:size())
        for i=1,nans:size(1) do
            if nans[i]:sum() > 0 then
                print ("Row",i,"has",nans[i]:sum(),"nans")
            end
        end
        return true
    end
end

function defaultdict(default_elem)
    local tbl = {}
    local mtbl = {}
    mtbl.__index = function(tbl, key)
        local val = rawget(tbl, key)
        return val or default_elem
    end
    setmetatable(tbl, mtbl)
    return tbl
end

function defaultdict_from_fxn(default_value_factory)
    local t = {}
    local metatable = {}
    metatable.__index = function(t, key)
        if not rawget(t, key) then
            rawset(t, key, default_value_factory(key))
        end
        return rawget(t, key)
    end
    return setmetatable(t, metatable)
end

function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

function spairs_by_value(T)
	-- decreasing order
	return spairs(T, function(t,a,b) return t[b] < t[a] end)
end

function test_spairs_by_value()
	a = {c=1, b=3, d=55,e=0,f=6,g=3}
	for k,v in spairs_by_value(a) do
		print(k,v)
	end
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function clone_list(tensor_list, zero_too)
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

table.reduce = function (list, fn) 
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = fn(acc, v)
        end 
    end 
    return acc 
end

function sum(t)
    return table.reduce(t,
        function (a, b)
            return a + b
        end
    )
end   

table.slice = function (list, first, last)
    local sliced = {}
    for i=first,last do
        table.insert(sliced, list[i])
    end
    return sliced
end

function where(bytes)
    assert(torch.isTypeOf(bytes, 'torch.ByteTensor') or torch.isTypeOf(bytes, 'torch.CudaTensor'), "Expected type torch.ByteTensor or torch.CudaTensor, got type " .. torch.type(bytes))
    out = {}
    for i=1,bytes:size(1) do
        if bytes[i] == 1 then
            out[#out + 1] = i
        end
    end
    return out
end