
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt 
import json
# perform qk calculation and get indices
# this version will not update in inference mode

def key_pruner_query_driven_prerope_2_q(kv_states, q_states, window_size=32, recent_size=0, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -window_size:, :], 2).mean(dim=2)
    q1 = queries_norm[..., : head_dim//2] 
    q2 = queries_norm[..., head_dim//2 : ] 
    q_score = torch.cat( (q1+q2, q1+q2), dim=-1 ) 

    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = q_score * keys_norm
    del q_states
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    del key
    mask = mask.scatter_(-1, keep_idx, 1)                   
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k).contiguous(), kv_states[:, :, seqlen - recent_size:, :], ~mask

def key_pruner_query_driven_prerope(kv_states, q_states, window_size=32, recent_size=0, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -window_size:, :], 2).mean(dim=2)
    q1 = queries_norm[..., : head_dim//2] 
    q2 = queries_norm[..., head_dim//2 : ] 
    q_score = torch.cat( (q1+q2, q1+q2), dim=-1 ) 

    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = q_score * keys_norm  
    del q_states
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    del key
    mask = mask.scatter_(-1, keep_idx, 1)                   
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k).contiguous(), kv_states[:, :, seqlen - recent_size:, :], ~mask

def key_pruner_query_driven(kv_states, q_states, window_size=32, recent_size=0, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -window_size:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    del q_states
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    del key
    mask = mask.scatter_(-1, keep_idx, 1)                   
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k).contiguous(), kv_states[:, :, seqlen - recent_size:, :], ~mask


def key_pruner_query_driven_with_anal(kv_states, q_states, window_size=32, recent_size=0, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -window_size:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    del q_states
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    del key
    mask = mask.scatter_(-1, keep_idx, 1)                   
    share_num = 4 
    mask_anal = mask.reshape(1, share_num, -1, head_dim) 
    mask_anal_sum = mask_anal.sum(dim=1).squeeze(0)
    #print(mask_anal_sum.shape)
    for i in range(mask_anal_sum.shape[0]) : 
        print( torch.unique(mask_anal_sum[i], dim=0, return_counts=True) )
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

def key_pruner_query_driven_dynamic(kv_states, q_states, window_size=32, recent_size=0, ratio=0.3):
    batch_size, head_num, seqlen, head_dim = kv_states.shape
    #print(kv_states.shape)
    k = int(head_dim * ratio *head_num)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -window_size:, :], 2).mean(dim=2) 
    # queries_mean = torch.mean(q_states[..., -window_size:, :], dim=2) 
    # queries_var = torch.std(q_states[..., -window_size:, :], dim=2) 
    # qv1 = queries_var[..., : head_dim//2] 
    # qv2 = queries_var[..., head_dim//2 : ] 
    # qv_score = torch.cat( (qv1+qv2, qv1+qv2), dim=-1 ).flatten()
    #print(queries_norm) 
    #print(queries_mean) 
    #print(queries_var)

    #print(queries_mean.shape)
    #print(queries_var.shape)
    q1 = queries_norm[..., : head_dim//2] 
    q2 = queries_norm[..., head_dim//2 : ] 
    q_score = torch.cat( (q1+q2, q1+q2), dim=-1 ).flatten()

    keys_norm = torch.pow(kv_states, 2).mean(dim=2).flatten() #[H, D]->[H*D]
    # print(q_score)
    # print(qv_score)
    score = (q_score) * keys_norm
    #print(q_score)
    del q_states
    _, indices = torch.topk(score, k, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(score.shape, dtype=torch.bool).to(kv_states.device)
    del score 
    kv_recent = kv_states[:, :, seqlen - recent_size:, :] 

    mask = mask.scatter_(-1, keep_idx, 1) #[H*D] -> [1, 1, H*D]
    #print(torch.sum(mask)/mask.numel())
    mask_k = mask.unsqueeze(0).unsqueeze(1).expand(-1, seqlen-recent_size, -1)
    kv_states = kv_states.transpose(1, 2).reshape(batch_size, seqlen, -1)
    return kv_states[:, :seqlen - recent_size, :][~mask_k].reshape(1, seqlen, -1), kv_recent, ~mask

def make_F_score(v, window_size=128) : 
    B, G, T, C = v.shape 
    v = v.permute(0, 2, 1, 3).reshape(B*T, G*C) 
    score = torch.norm(v[-window_size:, :], p='fro', dim=0)

    return score

def cos_sim(base_tensor, vector, layer_idx) : 
    B, G, T, C = base_tensor.shape 
    reshape_tensor = base_tensor.transpose(0, 2).reshape(T, -1) 
    vector = vector.flatten() 
    len_tensor = torch.norm(reshape_tensor, dim=1) 
    len_vector = torch.norm(vector**2)
    dot_products = (reshape_tensor * vector).sum(dim=1)
    cosine_sim = dot_products / (len_tensor * len_vector)
    similarity = torch.abs(cosine_sim).mean().item()
    file_path = './sim_pre.json'
    result = {"layer": layer_idx, "token": T, "similarity": similarity}
    with open(file_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    #print(similarity.item())
    return similarity
    #print(similarity)

def query_shot(q, T, layer_idx) : 
    print(q.shape) 
    #print(q[0, 0, 0, :])
    query_vector = q[0, 0, 0, :].tolist()
    file_path = './pre_key_shot.json'
    result = {"layer": layer_idx, "token": T, "q": query_vector} 
    if(layer_idx == 0) : 
        with open(file_path, "a") as f:
            f.write(json.dumps(result) + "\n")

def tensor_shot(tensor, T, layer_idx) : 
    print(tensor.shape) 
    abs_tensor = torch.abs(tensor[0, 0, :, :]).cpu()

    plt.figure(figsize=(12, 8))
    plt.imshow(abs_tensor, aspect='auto', cmap='gray_r')
    plt.xlabel('Prefill Key Dimension')
    plt.ylabel('Token Number')
    #title_str = f"Layer {layer}: Query Vector Magnitude" if layer is not None else "Query Vector Magnitude"
    #plt.title(title_str)
    plt.colorbar(label='Magnitude')
    plt.show()

def get_c_ratio(k) :  
    B, G, T, C = k.shape 
    total_count = k.numel()/2
    ratio = torch.abs( k[..., : C//2] / k[..., C//2 : ] )  
    print(ratio.shape)
    c_score = torch.where(ratio<1, ratio, 1/ratio)

    count_0_025   = ((c_score >= 0.0)   & (c_score < 0.25)).sum().item()
    count_025_05  = ((c_score >= 0.25)  & (c_score < 0.5)).sum().item()
    count_05_075  = ((c_score >= 0.5)   & (c_score < 0.75)).sum().item()
    count_075_1   = ((c_score >= 0.75)  & (c_score <= 1.0)).sum().item()

    ratio_0_025   = count_0_025  / total_count
    ratio_025_05  = count_025_05 / total_count
    ratio_05_075  = count_05_075 / total_count
    ratio_075_1   = count_075_1  / total_count

    print("r [0, 0.25):", ratio_0_025)
    print("r [0.25, 0.5):", ratio_025_05)
    print("r [0.5, 0.75):", ratio_05_075)
    print("r [0.75, 1.0]:", ratio_075_1)

    exit()

def make_F_score_select(v, window_size=128) : 
    B, G, T, C = v.shape 
    v = v.permute(0, 2, 1, 3).reshape(B*T, G*C) 
    score = torch.norm(v[-window_size:-1, :], p='fro', dim=0)
    return score

def optimal_Frobenius_prune(q, k, ratio) : 
    B, G, T, C = k.shape 
    #print(k.shape) 
    #k = k.permute(0, 2, 1, 3).reshape(B*T, G*C) 
    pruning_num = int(G*C*ratio) 
    
    k_score = make_F_score(k, window_size = 128) #[G*C] 
    q_score = make_F_score(q, window_size = 128) #[S*G*C]
    #print(k.shape)
    k = k.permute(0, 2, 1, 3).reshape(B*T, G*C) 
    pruned_k = torch.zeros_like(k)
    #print(pruned_k.shape)
    Score_metric = k_score * q_score 
    remaining_channels = torch.topk(Score_metric, pruning_num)[1]  # Shape: (pruning_num)

    pruned_k[ :, remaining_channels]  = k[ :, remaining_channels] 
    pruned_k = torch.reshape(pruned_k, (B, T, G, C)).transpose(1, 2)
    #print(pruned_k.shape)
    #exit()
    #del(k)
    return pruned_k

def anal_RoPE_dist(pre, post) : 
    B, G, T, C = pre.shape 
    pre = pre.permute(0, 2, 1, 3).reshape(B*T, G*C) 
    post = post.permute(0, 2, 1, 3).reshape(B*T, G*C) 

    pre_score = torch.norm(pre, p='fro', dim=0).sort()[0]
    post_score = torch.norm(post, p='fro', dim=0).sort()[0]
    
    #print(pre_score.shape) 
    #print(pre_score)
    plt.plot(pre_score[:256].cpu(), color='blue', label='pre-RoPE') 
    plt.plot(post_score[:256].cpu(), color='red', label='post-RoPE') 
    plt.legend()
    plt.show()
    #exit()

def draw_channel(pre, post, interval=4):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))   
    ax1.plot(torch.abs(pre[::interval]).cpu(), color='blue')
    ax1.set_title('Pre-RoPE')
    ax1.set_ylim(0, 4)   
    ax2.plot(torch.abs(post[::interval]).cpu(), color='red')
    ax2.set_title('Post-RoPE')
    ax2.set_ylim(0, 4)   
    
    plt.tight_layout()  
    plt.show()


def anal_RoPE_dist_with_query(q, pre, post) : 
    p = 0.4 
    B, G, T, C = pre.shape 
    print(pre.shape, post.shape)
    pruning_num = int(G*C*p)
    q_score = make_F_score(q, window_size=32) 
    pre_score = make_F_score(pre, window_size=T-1) 
    post_score = make_F_score(post, window_size=T-1) 
    print("post score", post_score)
    pre_rope_channel_loss = pre_score.sort()[0][:pruning_num].sum()
    post_rope_channel_loss = post_score.sort()[0][:pruning_num].sum()

    print("pre_rope_channel_loss", pre_rope_channel_loss)
    print("post_rope_channel_loss", post_rope_channel_loss)

    pre_score_with_query = (q_score*pre_score).sort()[0]
    post_score_with_query = (q_score*post_score).sort()[0] 
    
    pre_rope_pruning_loss = pre_score_with_query[:pruning_num].sum()
    post_rope_pruning_loss = post_score_with_query[:pruning_num].sum()

    # print("pre rope pruning loss", pre_rope_pruning_loss)
    # print("post rope pruning loss", post_rope_pruning_loss)
    # plt.plot(pre_score_with_query[:pruning_num].cpu(), color='blue', label='pre-RoPE') 
    # plt.plot(post_score_with_query[:pruning_num].cpu(), color='red', label='post-RoPE') 
    # plt.legend()
    # plt.show()
    #exit()
