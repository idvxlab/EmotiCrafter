import torch
from transformers import GPT2Model, GPT2Config
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block, GPT2Attention, GPT2MLP
)
from torch import nn

class Cond_Attention(GPT2Attention):
    def __init__(self, nx, n_ctx, config, is_cross_attention=False):
        super(GPT2Attention, self).__init__()
        self.output_attentions = config.output_attentions
        n_state = nx
        assert n_state % config.n_head == 0
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = n_state
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()
        self.c_z = Conv1D(n_state * 2, nx)
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self, x, z, layer_past=None, attention_mask=None, head_mask=None, use_cache=True, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache:
            present = (key, value)
        else:
            present = None

        z_conv = self.c_z(z)
        key_z, value_z = z_conv.split(self.split_size, dim=2)
        key_z = self._split_heads(key_z, self.num_heads, self.head_dim)
        value_z = self._split_heads(value_z, self.num_heads, self.head_dim)

        key = key_z 
        value = value_z
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Cond_Block(GPT2Block):
    def __init__(self, config,activate_a = False,activate_v = False):
        super(GPT2Block, self).__init__()
        self.activate_a = activate_a
        self.activate_v = activate_v
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        
        self.attn =  Cond_Attention(nx,config.n_ctx,config)
        
        self.attn_a =None if not self.activate_a else Cond_Attention(nx,config.n_ctx,config)
        self.ln_a = None if not self.activate_a else nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        
        self.attn_v =None if not self.activate_v else Cond_Attention(nx,config.n_ctx,config)
        self.ln_v = None if not self.activate_v else nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4 * nx, config)

    def forward(self, x, a,v, layer_past=None, attention_mask=None, head_mask=None):
        residual = x
        x = self.ln_1(x)
        attn_outputs = self.attn(
            x=x,
            z=x
        )
        attn_output = attn_outputs[0]
        # outputs = attn_outputs[1:]
        x = x + attn_output
        if self.activate_a:
            x = self.ln_a(x)
            cross_attn_outputs = self.attn_a(
                x=x,
                z=a
            )
            cross_attn_output = cross_attn_outputs[0]
            x = x + cross_attn_output
        if self.activate_v:
            x = self.ln_v(x)
            cross_attn_outputs = self.attn_v(
                x=x,
                z=v
            )
            cross_attn_output = cross_attn_outputs[0]
            x = x + cross_attn_output
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = (x,)
        return outputs
    
class EmotionInjectionTransformer(GPT2Model):
    def __init__(self, config, final_out_type="Linear+LN",sd_feature_dim=2048):
        super(GPT2Model, self).__init__(config)
        self.add_attn = True
        self.sd_feature_dim = sd_feature_dim
        self.activate_a = True 
        self.activate_v = True 
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.xl_feature2gpt_feature = nn.Linear(self.sd_feature_dim,config.n_embd,bias=False)
        self.gpt_feature2xl_feature = nn.Linear(config.n_embd,self.sd_feature_dim,bias=False)
        if final_out_type == "Linear+LN" or final_out_type=="Linear+LN+noResidual":
            self.ln_xl_feature = nn.LayerNorm(self.sd_feature_dim, eps=1e-5)
        elif final_out_type == "Linear+LN+Linear" or final_out_type=="Linear+LN+Linear+noResidual":
            self.ln_xl_feature = nn.LayerNorm(self.sd_feature_dim, eps=1e-5)
            self.ff = nn.Linear(self.sd_feature_dim,self.sd_feature_dim,bias=False)
        else:
            raise NotImplementedError
        self.init_weights()
        self.cross_token = 16
        self.a_f = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, config.n_embd*self.cross_token if self.activate_a else config.n_embd)
        )
        self.v_f = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, config.n_embd*self.cross_token if self.activate_v else config.n_embd)
        )
        if self.add_attn:
            self.attn_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.h = nn.ModuleList([Cond_Block(config,self.activate_a,self.activate_v) for _ in range(config.n_layer)])
        else:
            self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.final_out_type = final_out_type
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            arousal=None,
            valence=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        else:
            residual = inputs_embeds 
            inputs_embeds = self.xl_feature2gpt_feature(inputs_embeds)
    
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)
        
        a_feature = self.attn_proj(self.a_f(arousal).view(-1, self.cross_token, self.config.n_embd) )
        v_feature = self.attn_proj(self.v_f(valence).view(-1, self.cross_token, self.config.n_embd) )
        
        output_shape = input_shape + (hidden_states.size(-1),)

        all_self_attentions = () if self.output_attentions else None
        all_hidden_states = () if self.output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states, a = a_feature,v = v_feature, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )
            hidden_states = outputs[0]
            if self.output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if self.use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        
        
        if self.final_out_type == "Linear+LN":
            hidden_states = residual+self.ln_xl_feature(self.gpt_feature2xl_feature(hidden_states))
        elif self.final_out_type == "Linear+LN+noResidual":
            hidden_states = self.ln_xl_feature(self.gpt_feature2xl_feature(hidden_states))
        elif self.final_out_type == "Linear+LN+Linear":
            hidden_states = residual+self.ff(self.ln_xl_feature(self.gpt_feature2xl_feature(hidden_states)))
        elif self.final_out_type == "Linear+LN+Linear+noResidual":
            hidden_states = self.ff(self.ln_xl_feature(self.gpt_feature2xl_feature(hidden_states)))
        elif self.final_out_type == "Linear+noResidual":
            hidden_states = self.gpt_feature2xl_feature(hidden_states)
        else:
            hidden_states = residual+self.gpt_feature2xl_feature(hidden_states)
            
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            attention_output_shape = input_shape[:-1] + (-1,) + all_self_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_self_attentions)
            outputs = outputs + (all_attentions,)

        return outputs



    
    