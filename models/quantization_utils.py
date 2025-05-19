import torch 

def quant_per_token(k: torch.FloatTensor, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    # ================== Get Scale & Zeros ===============
    # Quantize
    k_reshape = k.permute(0, 2, 1, 3).reshape(B, T, -1)
    #new_shape = (B, nh, num_groups, group_size, D) 
    max_int = 2 ** bits - 1
    mn = torch.min(k_reshape, dim=-1, keepdim=True)[0]
    mx = torch.max(k_reshape, dim=-1, keepdim=True)[0]
    scale =  (mx - mn) / max_int
    data = k_reshape - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(B, T, nh, D).transpose(1, 2)
    #print(data)
    return data, scale, mn

def dequant_per_token(q: torch.IntTensor, scale: torch.FloatTensor, mn: torch.FloatTensor, bits: int):
    B, nh, T, D = q.shape
    q_reshape = q.permute(0, 2, 1, 3).reshape(B, T, nh * D)
    dequant = q_reshape * scale + mn  # (B, T, nh*D)
    dequant = dequant.view(B, T, nh, D).permute(0, 2, 1, 3)

    return dequant

def fake_quant_data_per_token(data, bit=3) : 
    bits = bit
    quant_tensor, scale, mn = quant_per_token(data, bits)
    dequant_tensor = dequant_per_token(quant_tensor, scale, mn, bits)
    return dequant_tensor 

def quant_per_token_TC(k: torch.FloatTensor, bits: int):
    assert len(k.shape) == 3
    shape = k.shape
    B, T, D = shape
    # ================== Get Scale & Zeros ===============
    # Quantize
    #new_shape = (B, nh, num_groups, group_size, D) 
    max_int = 2 ** bits - 1
    mn = torch.min(k, dim=-1, keepdim=True)[0]
    mx = torch.max(k, dim=-1, keepdim=True)[0]
    scale =  (mx - mn) / max_int 
    data = k - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    #data = data.view(B, T, nh, D).transpose(1, 2)
    return data, scale, mn

def dequant_per_token_TC(q: torch.IntTensor, scale: torch.FloatTensor, mn: torch.FloatTensor):
    #B, nh, T, D = q.shape
    #q_reshape = q.permute(0, 2, 1, 3).reshape(B, T, nh * D)
    dequant = q * scale + mn  # (B, T, nh*D)
    #dequant = dequant.view(B, T, nh, D).permute(0, 2, 1, 3)

    return dequant

def fake_quant_per_token_TC(data, bits=3) :
    quant, scale, mn = quant_per_token_TC(data, bits)
    dequant_tensor = dequant_per_token_TC(quant, scale, mn) 

    return dequant_tensor 

if __name__=="__main__" : 
    B, G, T, C = 1, 2, 4, 8
    testing_tensor = torch.randn( B, T, C ) 
    print(testing_tensor)
    bits = 8
   
    quant, scale, mn = quant_per_token_TC(testing_tensor, bits)
    dequant_tensor = dequant_per_token_TC(quant, scale, mn)
    
    print(dequant_tensor)
    print(torch.isclose(testing_tensor, dequant_tensor, 1e-2)) 
    # print(s.shape)
    # print(mn.shape)




def quant_per_channel(k: torch.FloatTensor, group_size: int, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B, nh, num_groups, group_size, D)
    # Quantize
    max_int = 2 ** bits - 1
    data = k.view(new_shape)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale =  (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)
    #print(data)
    return data, scale, mn

def dequant_per_channel(data: torch.FloatTensor, 
                        scale: torch.FloatTensor, 
                        mn: torch.FloatTensor,
                        group_size: int, 
                        bits: int,
                        ):
    pack_dim = 2
    #assert bits in [2, 4, 8]
    assert len(data.shape) == 4
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(shape[:pack_dim] + (num_groups, group_size,) + shape[pack_dim+1:])
    data = data.to(torch.float16)
    data = data * scale + mn 
    return data.view(shape)

def fake_quant_data_per_channel(data, bit=3) : 
    B, G, T, C = data.shape 
    bits = bit
    quant_tensor, scale, mn = quant_per_channel(data, T, bits)
    dequant_tensor = dequant_per_channel(quant_tensor, scale, mn, T, bits)
    return dequant_tensor 

def fake_quant_data_return_param(data, bit=3) : 
    B, G, T, C = data.shape 
    bits = bit
    quant_tensor, scale, mn = quant_per_channel(data, T, bits)
    dequant_tensor = dequant_per_channel(quant_tensor, scale, mn, T, bits)
    return dequant_tensor, scale, mn

def get_scale(k: torch.FloatTensor, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    # ================== Get Scale & Zeros ===============
    #assert T % group_size == 0
    #num_groups = T // group_size
    #new_shape = (B, nh, num_groups, group_size, D)
    # Quantize
    max_int = 2 ** bits - 1
    #data = k.view(new_shape)
    mn = torch.min(k, dim=-2, keepdim=True)[0]
    mx = torch.max(k, dim=-2, keepdim=True)[0]
    scale =  (mx - mn) / max_int
    #data = data - mn
    #data.div_(scale)
    #data = data.clamp_(0, max_int).round_().to(torch.int32)
    #data = data.view(shape)
    #print(data)
    return scale, mn

def get_scale_TC(data: torch.FloatTensor, bits: int):
    assert len(data.shape) == 3
    B, T, D = data.shape
    shape = data.shape
    # ================== Get Scale & Zeros ===============
    #assert T % group_size == 0
    #num_groups = T // group_size
    #new_shape = (num_groups, group_size, D)
    
    # Quantize
    max_int = 2 ** bits - 1
    #data = k.view(new_shape)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale =  (mx - mn) / max_int
    #data = data - mn
    #data.div_(scale)
    #data = data.clamp_(0, max_int).round_().to(torch.int32)
    #data = data.view(shape)
    #print(data)
    return scale, mn

def quant_per_channel_TC(data: torch.FloatTensor, 
                        bits : int
                        ):
    
    assert len(data.shape) == 3
    B, T, D = data.shape
    shape = data.shape
    max_int = 2 ** bits - 1
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale =  (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    return data, scale, mn


def dequant_per_channel_TC(data: torch.FloatTensor, 
                        scale: torch.FloatTensor, 
                        mn: torch.FloatTensor                        
                        ):
    data = data.to(torch.float16)
    data = data * scale + mn 
    return data


def fake_quant_data_TC(data,  bits : int) : 
    B, T, C = data.shape 
    quant_tensor, scale, mn = quant_per_channel_TC(data, bits)
    dequant_tensor = dequant_per_channel_TC(quant_tensor, scale, mn)
    return dequant_tensor, scale, mn  

def fake_quant_with_param(data, scale, mn, bits) : 
    max_int = 2 ** bits - 1
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_() 
    data = data * scale + mn  

    return data 

