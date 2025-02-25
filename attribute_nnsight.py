# %load_ext autoreload
# %autoreload 2

from functools import partial

import torch
from transformer_lens import HookedTransformer

from graph import Graph
from attribute import attribute
from dataset import HFEAPDataset
from metrics import get_metric
from evaluate_graph import evaluate_graph, evaluate_baseline, evaluate_area_under_curve

from nnsight import LanguageModel

from typing import Callable, List, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from tqdm import tqdm
from einops import einsum

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode


def to_tokens(tokenizer, model, input_text, prepend_bos=True, padding_side='right', move_to_device=True, truncate=True, max_length=None):
    """
    Converts input text to tokens using HuggingFace tokenizer with similar functionality to TransformerLens.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (for device information if move_to_device=True)
        input_text (Union[str, List[str]]): The input to tokenize
        prepend_bos (bool, optional): Whether to prepend the BOS token. Defaults to True.
        padding_side (str, optional): Side to pad on ('left' or 'right'). Defaults to 'right'.
        move_to_device (bool, optional): Whether to move tensors to model's device. Defaults to True.
        truncate (bool, optional): Whether to truncate to model's max length. Defaults to True.
        max_length (int, optional): Maximum length to truncate to. If None, uses model's max length.
        
    Returns:
        torch.Tensor: Tensor of token ids
    """
    # Save original padding side
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    
    # Handle BOS token
    add_special_tokens = prepend_bos
    
    # Determine max_length for truncation
    if truncate and max_length is None:
        if hasattr(model.config, 'max_position_embeddings'):
            max_length = model.config.max_position_embeddings
        else:
            max_length = model.config.n_positions  # for GPT-2
    
    # Tokenize
    tokens = tokenizer(
        input_text,
        add_special_tokens=add_special_tokens,  # This handles BOS token if model has one
        padding=True if isinstance(input_text, list) else False,  # Only pad for batch inputs
        truncation=truncate,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device if requested
    if move_to_device and hasattr(model, 'device'):
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    return tokens['input_ids']

# def tokenize_plus_nnsight(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
#     """
#     Tokenizes the input strings using the provided model.

#     Args:
#         model (HookedTransformer): The model used for tokenization.
#         inputs (List[str]): The list of input strings to be tokenized.

#     Returns:
#         tuple: A tuple containing the following elements:
#             - tokens (torch.Tensor): The tokenized inputs.
#             - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
#             - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
#             - n_pos (int): The maximum sequence length of the tokenized inputs.
#     """
#     if max_length is not None:
#         old_n_ctx = model.config.n_ctx
#         model.config.n_ctx = max_length


#     # tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
#     # Shun's change
#     tokenizer = model.tokenizer
#     tokens = to_tokens(tokenizer, model, inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    
    
#     if max_length is not None:
#         model.config.n_ctx = old_n_ctx
#     attention_mask = get_attention_mask(model.tokenizer, tokens, True)
#     input_lengths = attention_mask.sum(1)
#     n_pos = attention_mask.size(1)
#     return tokens, attention_mask, input_lengths, n_pos

def tokenize_plus_nnsight(model, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                       intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                       intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    """Gets edge attribution scores using EAP with nnsight.
    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cpu', dtype=model.dtype)

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations_nns(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus_nns(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus_nns(model, corrupted)

        activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.config.hidden_size), 
                                         device=model.device, dtype=model.dtype)

        with torch.inference_mode():
            if intervention == 'patching':
                # Get corrupted activations
                with model.trace({"input_ids": corrupted_tokens, "attention_mask": attention_mask}):
                    # Input embeddings
                    node = graph.nodes['input']
                    fwd_index = graph.forward_index(node)
                    activation_difference[:, :, fwd_index] += model.transformer.wte.output

                    for layer in range(graph.cfg['n_layers']):
                        # Attention heads
                        node = graph.nodes[f'a{layer}.h0']
                        fwd_index = graph.forward_index(node)
                        
                        attn_hs = model.transformer.h[layer].attn.c_proj.input
                        attn_hs = attn_hs.view(attn_hs.shape[0], attn_hs.shape[1], 
                                             model.config.n_head, model.config.hidden_size // model.config.n_head)
                        
                        W_O = model.transformer.h[layer].attn.c_proj.weight.view(
                            model.config.n_head, model.config.hidden_size // model.config.n_head, model.config.hidden_size)
                        
                        by_head = einsum(attn_hs, W_O, 
                                       "batch pos head_idx head_dim, head_idx head_dim model_dim -> batch pos head_idx model_dim")
                        
                        activation_difference[:, :, fwd_index] += by_head

                        # MLP
                        node = graph.nodes[f'm{layer}']
                        fwd_index = graph.forward_index(node)
                        activation_difference[:, :, fwd_index] += model.transformer.h[layer].mlp.output
                    
            elif 'mean' in intervention:
                activation_difference += means

            # Get clean logits
            clean_logits = model.trace({'input_ids': clean_tokens, 
                                      'attention_mask': attention_mask}, 
                                     trace=False)['logits']

            # Get clean activations and compute gradients
            with model.trace({"input_ids": clean_tokens, "attention_mask": attention_mask}):
                # Input embeddings
                node = graph.nodes['input']
                fwd_index = graph.forward_index(node)
                activation_difference[:, :, fwd_index] -= model.transformer.wte.output

                for layer in range(graph.cfg['n_layers']):
                    # Process QKV inputs
                    if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.config.n_head)):
                        node = graph.nodes[f'a{layer}.h0']
                        prev_index = graph.prev_index(node)
                        bwd_index = graph.backward_index(node, qkv='q', attn_slice=False)
                        bwd_index = slice(bwd_index, bwd_index + 3 * model.config.n_head)
                        
                        # Update scores based on QKV differences
                        if activation_difference.shape[-1] == model.config.hidden_size:
                            grads = model.transformer.h[layer].ln_1.input.grad
                            grads = grads.unsqueeze(2)  # Add head dimension
                            s = einsum(activation_difference[:, :, :prev_index], grads,
                                     'batch pos forward hidden, batch pos backward hidden -> forward backward')
                            scores[:prev_index, bwd_index] += s.squeeze(1)

                    # Process attention outputs
                    node = graph.nodes[f'a{layer}.h0']
                    fwd_index = graph.forward_index(node)
                    
                    attn_hs = model.transformer.h[layer].attn.c_proj.input
                    attn_hs = attn_hs.view(attn_hs.shape[0], attn_hs.shape[1], 
                                         model.config.n_head, model.config.hidden_size // model.config.n_head)
                    
                    W_O = model.transformer.h[layer].attn.c_proj.weight.view(
                        model.config.n_head, model.config.hidden_size // model.config.n_head, model.config.hidden_size)
                    
                    by_head = einsum(attn_hs, W_O, 
                                   "batch pos head_idx head_dim, head_idx head_dim model_dim -> batch pos head_idx model_dim")
                    
                    activation_difference[:, :, fwd_index] -= by_head

                    # Process MLP
                    if graph.nodes[f'm{layer}'].in_graph:
                        node = graph.nodes[f'm{layer}']
                        prev_index = graph.prev_index(node)
                        bwd_index = graph.backward_index(node)
                        
                        # Update scores based on MLP differences
                        grads = model.transformer.h[layer].ln_2.input.grad
                        s = einsum(activation_difference[:, :, :prev_index], grads,
                                 'batch pos forward hidden, batch pos backward hidden -> forward backward')
                        scores[:prev_index, bwd_index] += s.squeeze(1)

                    node = graph.nodes[f'm{layer}']
                    fwd_index = graph.forward_index(node)
                    activation_difference[:, :, fwd_index] -= model.transformer.h[layer].mlp.output

                # Get logits and compute metric
                logits = model.lm_head.output.save()
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    return scores

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor]):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input). The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
    """
    separate_activations = model.config.use_normalization_before_and_after and scores is None
    if separate_activations:
        activation_difference = torch.zeros((2, batch_size, n_pos, graph.n_forward, model.config.d_model), device=model.config.device, dtype=model.config.dtype)
    else:
        # activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.config.d_model), device=model.config.device, dtype=model.config.dtype)
        activation_difference = torch.zeros(
            (batch_size, n_pos, graph.n_forward, model.config.n_embd),  # using n_embd instead of d_model
            device=model.device,  # device from model instead of config
            dtype=model.dtype    # dtype from model instead of config
        )

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
        
    # Fills up the activation difference matrix. In the default case (not separate_activations), 
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach()
        try:
            if separate_activations:
                if add:
                    activation_difference[0, :, :, index] += acts
                else:
                    activation_difference[1, :, :, index] += acts
            else:
                if add:
                    activation_difference[:, :, index] += acts
                else:
                    activation_difference[:, :, index] -= acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass 
            hook (_type_): (unused)

        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
            print(prev_index, bwd_index, scores.size(), s.size())
            raise e
    
    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
    
    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate('qkv'):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
        
    node = graph.nodes['logits']
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference



def get_scores_eap_nnsight(model: LanguageModel, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cpu', dtype=model.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        
        # clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus_nnsight(model, clean)
        # corrupted_tokens, _, _, _ = tokenize_plus_nnsight(model, corrupted)

        # Convert tokens to real tensors
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus_nnsight(model, clean)
        clean_tokens = clean_tokens.clone().detach()  # Ensure real tensor
        attention_mask = attention_mask.clone().detach()
        
        corrupted_tokens, _, _, _ = tokenize_plus_nnsight(model, corrupted)
        corrupted_tokens = corrupted_tokens.clone().detach()  # Ensure real tensor

        with torch.inference_mode():
            if intervention == 'patching':
                activation_difference = torch.zeros(
                    (batch_size, n_pos, graph.n_forward, model.config.hidden_size), 
                    device=model.device, 
                    dtype=model.dtype
                )

                # Get clean logits first using trace=False
                # clean_logits = model.trace({'input_ids': clean_tokens, 'attention_mask':attention_mask}, trace=False)['logits']
                
                print(f"corrupted_tokens is {corrupted_tokens}")
                print(f"attention_mask is {attention_mask}")
                # Capture corrupted activations
                with model.trace({"input_ids": corrupted_tokens, "attention_mask": attention_mask}):
                    for layer in range(graph.cfg['n_layers']):
                        node = graph.nodes[f'a{layer}.h0']
                        fwd_index = graph.forward_index(node)
                        attn_hs = model.transformer.h[layer].attn.c_proj.input

                        by_head = split_heads_nns(attn_hs, model.config.n_head, model.config.hidden_size)

                        by_head = model.transformer.h[layer].attn.c_proj(by_head)
                        by_head = model.transformer.h[layer].attn.resid_dropout(by_head)
                        
                        activation_difference[:, :, fwd_index] += by_head

                        node = graph.nodes[f'm{layer}']
                        fwd_index = graph.forward_index(node)
                        activation_difference[:, :, fwd_index][:] += model.transformer.h[layer].mlp.output[:]

            elif 'mean' in intervention:
                activation_difference += means

            # Run with input modifications
            with model.trace({"input_ids": clean_tokens, "attention_mask": attention_mask}):
                for layer in range(graph.cfg['n_layers']):
                    if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.config.n_head)):
                        # Get layer input
                        qkv_inp = model.transformer.h[layer].ln_1.input
                        node = graph.nodes[f'a{layer}.h0']
                        fwd_index = graph.forward_index(node)
                        
                        update = activation_difference[:, :, fwd_index]
                        model.transformer.h[layer].attn.output[:] += update

                    # MLP handling
                    if graph.nodes[f'm{layer}'].in_graph:
                        node = graph.nodes[f'm{layer}']
                        fwd_index = graph.forward_index(node)
                        model.transformer.h[layer].mlp.output[:] += activation_difference[:, :, fwd_index]

                logits = model.lm_head.output.save()
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

                # Collect gradients
                grads = model.lm_head.output.grad
                s = einsum(activation_difference, grads,
                          'batch pos forward hidden, batch pos token -> forward')
                scores += s

    scores /= total_items
    return scores

allowed_aggregations = {'sum', 'mean'}#, 'l2'}        
def attribute_nnsight(model: LanguageModel, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], method: Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations'], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', aggregation='sum', ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    assert model.config.use_attn_result, "Model must be configured to use attention result (model.config.use_attn_result)"
    assert model.config.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.config.use_split_qkv_input)"
    assert model.config.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.config.use_hook_mlp_in)"
    if model.config.n_key_value_heads is not None:
        assert model.config.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.config.ungroup_grouped_query_attention = True)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap_nnsight(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader, quiet=quiet)
    elif method == 'EAP-IG-inputs':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'clean-corrupted':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention, intervention_dataloader=intervention_dataloader, quiet=quiet)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations'], but got {method}")


    if aggregation == 'mean':
        scores /= model.config.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)


# model = LanguageModel('gpt2', device_map='cpu')
# model.config.use_split_qkv_input = True
# model.config.use_attn_result = True
# model.config.use_hook_mlp_in = True
# model.config.ungroup_grouped_query_attention = True

# dataset = HFEAPDataset("danaarad/ioi_dataset", model.tokenizer, task="ioi", num_examples=100)
# dataloader = dataset.to_dataloader(20)
# metric_fn = get_metric("logit_diff", "ioi", model.tokenizer, model)

# model.config.n_key_value_heads = None
# model.config.dtype = torch.float32
# model.config.use_normalization_before_and_after = False

# g = Graph.from_model(model)


# model = LanguageModel('gpt2', device_map='cpu', dispatch='True')
model = LanguageModel('gpt2', device_map='cpu')

model.config.use_split_qkv_input = True
model.config.use_attn_result = True
model.config.use_hook_mlp_in = True
model.config.ungroup_grouped_query_attention = True

dataset = HFEAPDataset("danaarad/ioi_dataset", model.tokenizer, task="ioi", num_examples=100)
dataloader = dataset.to_dataloader(20)
metric_fn = get_metric("logit_diff", "ioi", model.tokenizer, model)

model.config.n_key_value_heads = None
model.config.dtype = torch.float32
model.config.use_normalization_before_and_after = False

g = Graph.from_model(model)

attribute_nnsight(model, g, dataloader, partial(metric_fn, loss=True, mean=True), 'EAP')