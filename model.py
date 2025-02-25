# model.py
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
)

def apply_rotary_pos_emb(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    sin, cos = rotary_emb[..., :head_dim // 2], rotary_emb[..., head_dim // 2:]
    rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated_x

class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        '''if seq_len <= 0:
            logging.warning(f"Invalid seq_len: {seq_len}, setting to 1")
            seq_len = 1
        if seq_len > self.max_position_embeddings:
            logging.warning(
                f"seq_len {seq_len} exceeds max_position_embeddings {self.max_position_embeddings}, truncating")
            seq_len = self.max_position_embeddings'''

        positions = torch.arange(seq_len, device=device)
        sincos = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)
        # Rearranged so that seq_len is in dimension 2
        return emb[None, None, :, :]

    def apply_rotary_emb(self, x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        # Wrapper to use the standalone function
        return apply_rotary_pos_emb(x, rotary_emb)


class MLP(nn.Module):  ###Inspired from LLamaMLP
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepSeekMoE(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 num_experts=8,
                 num_shared_experts=1,
                 top_k_experts=2):
        super().__init__()

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_attention_heads = num_attention_heads
        self.top_k_experts = top_k_experts

        self.num_routed_experts = num_experts - num_shared_experts

        # shared experts
        self.shared_experts = nn.ModuleList([
            MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
            for _ in range(self.num_shared_experts)
        ])

        # routed experts
        self.routed_experts = nn.ModuleList([
            MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        # Router components
        self.router = nn.Linear(hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))

    def forward(self, x):
        batch, seq_len, hidden_size = x.shape

        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts

        routing_logits = self.router(x) + self.routing_bias

        # Get top-k experts per token
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k_experts, dim=-1)

        # Normalize top-k scores
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Combined output
        combined_output = torch.zeros_like(x)

        for k in range(self.top_k_experts):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k + 1]

            # process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)

                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    combined_output[mask] += expert_output * expert_scores[mask]

        # Final output
        final_output = combined_output + shared_output
        return final_output

    def update_bias_terms(self, expert_load):
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load

        update_rate = 0.1 * torch.abs(load_diff)

        self.routing_bias.data -= update_rate * load_diff


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, seq_len, compression_ratio=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        self.latent_dim = self.hidden_size // compression_ratio

        # self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        # self.attention_dropout = attention_dropout
        self.is_causal = True

        # RoPE Emeddings : Half size for RoPE components
        self.rotary_emb = DeepseekRotaryEmbedding(self.head_dim//2)   #internally it will be applied to head_dim//2
        # Query, Key, Value projections
        # self.q_proj = nn.Linear(hidden_size, self.head_dim * num_attention_heads, bias=False)
        # self.k_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)
        # self.v_proj = nn.Linear(hidden_size, self.head_dim * num_key_value_heads, bias=False)
        # self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        ## Compressed KV Projection
        self.kv_proj_d = nn.Linear(hidden_size, self.latent_dim, bias=False)
        self.q_proj_d = nn.Linear(hidden_size, self.latent_dim, bias=False)

        self.k_proj_u = nn.Linear(self.latent_dim, hidden_size // 2, bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, hidden_size // 2, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, hidden_size, bias=False)

        self.rope_k = nn.Linear(self.hidden_size, hidden_size // 2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, hidden_size // 2, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, position_embeddings=None):
        batch, seq_len, _ = hidden_states.shape
        # hidden_shape = (batch, seq_len, -1, self.head_dim)

        kv_d = self.kv_proj_d(hidden_states)  # [ B, seq_len, latent_dim ]
        q_d = self.q_proj_d(hidden_states)  # [ B, seq_len, latent_dim ]

        k_proj_2 = self.k_proj_u(kv_d)  # [ B, seq_len, hidden/2 ]
        q_proj_2 = self.q_proj_u(q_d)  # [ B, seq_len, hidden/2 ]
        v_proj = self.v_proj_u(kv_d)  # [ B, seq_len, hidden ]

        k_rope_2 = self.rope_k(hidden_states)  # [ B, seq_len, hidden/2 ]
        q_rope_2 = self.rope_q(q_d)  # [ B, seq_len, hidden/2 ]

        #print('q_rope_2.shape, k_rope_2.shape: ', q_rope_2.shape, k_rope_2.shape)
        # Reshape components for heads before RoPE

        k_proj_2 = k_proj_2.view(batch, seq_len, self.num_attention_heads, self.head_dim // 2)
        k_rope_2 = k_rope_2.view(batch, seq_len, self.num_attention_heads, self.head_dim // 2)
        q_proj_2 = q_proj_2.view(batch, seq_len, self.num_attention_heads, self.head_dim // 2)
        q_rope_2 = q_rope_2.view(batch, seq_len, self.num_attention_heads, self.head_dim // 2)

        # Transpose to (batch, num_heads, seq_len, head_dim//2) for rotary application.
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)

        # Apply RoPE to KQ

        # Generate rotary embeddings. This returns a tensor of shape (1, 1, seq_len, head_dim//2).
        rotary_emb_out = self.rotary_emb(seq_len, hidden_states.device)

        #print(cos.shape, sin.shape)

        q_rope_2 = self.rotary_emb.apply_rotary_emb(q_rope_2, rotary_emb_out)
        k_rope_2 = self.rotary_emb.apply_rotary_emb(k_rope_2, rotary_emb_out)

        # Transpose back to (batch, seq_len, num_heads, head_dim//2)
        k_rope_2 = k_rope_2.transpose(1, 2)
        q_rope_2 = q_rope_2.transpose(1, 2)

        k = torch.cat([k_proj_2, k_rope_2], dim=-1)
        q = torch.cat([q_proj_2, q_rope_2], dim=-1)
        v = v_proj.view(batch, seq_len, self.num_attention_heads, self.head_dim)

        # Reshape
        q = q.transpose(1, 2)       #[ B, heads, seq_len, head_dim ]
        k = k.transpose(1, 2)       #[ B, heads, seq_len, head_dim ]
        v = v.transpose(1, 2)   # [ B, heads, seq_len, head_dim ]

        # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v,
                                           is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len,
                                                self.hidden_size)  # re-assemble all head outputs side by side

        # output projection
        y = self.o_proj(y)
        return y


class DeepseekTransformerBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 num_experts,
                 num_shared_experts,
                 top_k_experts,
                 compression_ratio,
                 eps,
                 seq_len):
        super(DeepseekTransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.seq_len = seq_len
        # self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == hidden_size, "Hidden size must be divisible by the number of attention heads."
        # assert self.hidden_size % self.num_key_value_heads == 0, "hidden_size must be divisible by num_key_value_heads"

        self.layer_norm_1 = LlamaRMSNorm(self.hidden_size, eps=eps)

        self.attn = MultiHeadLatentAttention(hidden_size, num_attention_heads, seq_len=seq_len,
                                             compression_ratio=compression_ratio)

        # Feedforward layer
        self.feed_forward = DeepSeekMoE(hidden_size,
                                        intermediate_size,
                                        num_attention_heads,
                                        num_experts=num_experts,
                                        num_shared_experts=num_shared_experts,
                                        top_k_experts=top_k_experts)

        self.layer_norm_2 = LlamaRMSNorm(self.hidden_size, eps=eps)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, position_embeddings=None):
        # Layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)

        '''
        # Query projection
        query = self.query_proj(hidden_states)
        query = query.view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads,
                           self.head_dim).transpose(1, 2)

        # Key and Value projections with shared num_key_value_heads
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)

        key = key.view(hidden_states.size(0), hidden_states.size(1), self.num_key_value_heads,
                       self.head_dim).transpose(1, 2)
        value = value.view(hidden_states.size(0), hidden_states.size(1), self.num_key_value_heads,
                           self.head_dim).transpose(1, 2)

        # Expand keys and values to match num_attention_heads
        key = key.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        value = value.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)

        # Apply rotary embeddings to query and key
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=True)

        # Reshape back to [batch_size, seq_length, hidden_size]
        attention_output = attention_output.transpose(1, 2).contiguous().view(hidden_states.size(0), -1,
                                                                              self.hidden_size)

        # Output projection
        attention_output = self.out_proj(attention_output)
        '''
        attention_output = self.attn(hidden_states, position_ids)

        # Residual connection
        hidden_states = residual + attention_output

        # Feedforward layer
        residual = hidden_states

        # Feed-forward
        hidden_states = self.layer_norm_2(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)

        hidden_states = residual + feed_forward_output

        return hidden_states


class CustomDeepSeekV3(nn.Module):
    def __init__(self, config):
        super(CustomDeepSeekV3, self).__init__()
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.seq_len = config['sequence_length']
        # self.num_key_value_heads = config['num_key_value_heads']
        # self.max_position_embeddings = config['max_position_embeddings']

        self.num_experts = config['num_experts']
        self.num_shared_experts = config['num_shared_experts']
        self.top_k_experts = config['top_k_experts']
        self.compression_ratio = config['compression_ratio']

        self.intermediate_size = config['intermediate_size']
        self.initializer_range = config['initializer_range']
        self.eps = config['rms_norm_eps']

        self.head_dim = self.hidden_size // self.num_attention_heads

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList([
            DeepseekTransformerBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.intermediate_size,
                self.num_experts,
                self.num_shared_experts,
                self.top_k_experts,
                self.compression_ratio,
                self.eps,
                self.seq_len
            ) for _ in range(self.num_hidden_layers)
        ])

        self.layer_norm = LlamaRMSNorm(self.hidden_size, eps=self.eps)

        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Share weights between embedding and lm_head
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(0, seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embeddings = self.embedding(input_ids)

        hidden_states = embeddings
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

