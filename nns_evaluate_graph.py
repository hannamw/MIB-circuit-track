from typing import Union, Callable, List, Literal, Optional

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from einops import einsum
from graph import Graph, AttentionNode

from tqdm import tqdm

from nns_attribute import tokenize_plus_nns
from nns_attribute import make_hooks_and_matrices_nns

import nnsight

def evaluate_graph_nns(
        model, 
        graph: Graph, 
        dataloader: DataLoader, 
        metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]], 
        quiet=False, 
        intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
        intervention_dataloader: Optional[DataLoader]=None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.

    Args:
        model (HookedTransformer): The model to run the circuit on 
        graph (Graph): The circuit to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): The metric(s) to evaluate with respect to
        quiet (bool, optional): Whether to silence the tqdm progress bar. Defaults to False.
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): Which ablation to evaluate with respect to. 'patching' is an interchange intervention; mean-positional takes the positional mean over the given dataset. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): The dataset to take the mean over. Must be set if intervention is mean or mean-positional. Defaults to None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of faithfulness scores; if a list, each list entry corresponds to a metric in the input list
    """
    #TODO
    #assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    #if model.cfg.n_key_value_heads is not None:
        #assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
        
    assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"
    
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        #TODO
        #means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    # This step cleans up the graph, removing components until it's fully connected
    graph.prune()

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = graph.in_graph.to(device=model.device, dtype=model.dtype)
    
    # same thing but for neurons
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device='cpu', dtype=torch.float16)

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.config.hidden_size).to(torch.float16)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use it as a mask to specify 
    # which edges we want to corrupt
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix
        
    #TODO Gemma 2
    """ if model.cfg.use_normalization_before_and_after:
        
        # If the model also normalizes the outputs of attention heads, we'll need to take that into account when evaluating the graph.
        attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device='cuda', dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks]) """


    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph 
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # Case where layernorm is applied after attention (gemma only)
            #TODO Gemma 2
            """ if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]
                
                # get the clean outputs of the attention heads that came before
                clean_attention_results = einsum(activation_matrix[1, :, :, :len(in_graph_vector)], attention_head_mask[:len(in_graph_vector)], 'batch pos previous hidden, previous layer -> batch pos layer hidden')
                
                # get the update corresponding to non-attention heads, and the difference between clean and corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector, non_attention_head_mask[:len(in_graph_vector)], 'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector, attention_head_mask[:len(in_graph_vector)], 'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden')                    
                else:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector, non_attention_head_mask[:len(in_graph_vector)], 'batch pos previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector, attention_head_mask[:len(in_graph_vector)], 'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden')
                
                # add the biases to the attention results, and compute the corrupted attention results using the difference
                # we process all the attention heads at once; this is how we can tell if we're doing that
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                
                # pass both the clean and corrupted attention results through the layernorm and 
                # add the difference to the update
                update = non_attention_update
                valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])                        
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i]) """
                        
            # In the non-gemma case, things are easy!
            activation_differences = activation_matrix
            # The ... here is to account for a potential head dimension, when constructing a whole attention layer's input
            if neuron_matrix is not None:
                update = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector,'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
            else:
                update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,'batch pos previous hidden, previous ... -> batch pos ... hidden')
            activations += update
            return activations
        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in the layer is in the graph, just construct the input for the entire layer
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))
                    
            # add MLP hook if MLP in graph
            if graph.nodes[f'm{layer}'].in_graph:
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))
                    
        # Always add the logits hook
        node = graph.nodes['logits']
        fwd_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
        input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks
    
    # convert metrics to list if it's not already
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]
    
    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)

    counter = 0
    save_data = {}
    for clean, corrupted, label in dataloader:
        counter += 1
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus_nns(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus_nns(model, corrupted)
        activation_difference = torch.zeros((len(clean), n_pos, graph.n_forward, model.config.hidden_size), device=model.device, dtype=model.dtype)

        # fwd_hooks_corrupted adds in corrupted acts to activation_difference
        # fwd_hooks_clean subtracts out clean acts from activation_difference
        # activation difference is of size (batch, pos, src_nodes, hidden)
        #(fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices_nns(model, graph, len(clean), n_pos, None)
        
        #input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.trace({"input_ids": corrupted_tokens, "attention_mask": attention_mask}):

                    node = graph.nodes['input']
                    fwd_index = graph.forward_index(node)
                    activation_difference[:, :, fwd_index] += model.transformer.wte.output

                    # if counter == 1:
                        #nnsight.apply(save_data.update, {f"embed": activation_difference.clone().detach()})

                    for layer in range(graph.cfg['n_layers']):
                        #### Attention #########
                        node = graph.nodes[f'a{layer}.h0']
                        fwd_index = graph.forward_index(node)
                        attn_hs = model.transformer.h[layer].attn.c_proj.input # [20, 20, 768]

                        attn_hs = attn_hs.view(attn_hs.shape[0], attn_hs.shape[1], model.config.n_head, model.config.hidden_size // model.config.n_head)

                        W_O = model.transformer.h[layer].attn.c_proj.weight.view(model.config.n_head, model.config.hidden_size // model.config.n_head, model.config.hidden_size)

                        by_head = einsum(attn_hs, W_O, "batch pos head_idx head_dim, head_idx head_dim model_dim -> batch pos head_idx model_dim")

                        #by_head = split_heads_nns(attn_hs, model.config.n_head, model.config.hidden_size) # [20, 20, 12, 768]

                        # by_head = model.transformer.h[layer].attn.c_proj(by_head) # [20, 20, 12, 768]

                        # by_head = by_head - model.transformer.h[layer].attn.c_proj.bias

                        activation_difference[:, :, fwd_index] += by_head

                        # if counter == 1:
                        #     nnsight.apply(save_data.update, {f"attn_{layer}": activation_difference.clone().detach()})

                        #### MLP #########
                        node = graph.nodes[f'm{layer}']
                        fwd_index = graph.forward_index(node)
                        activation_difference[:, :, fwd_index][:] += model.transformer.h[layer].mlp.output[:]

                        if counter == 1:
                            nnsight.apply(save_data.update, {f"mlp_{layer}": activation_difference.clone().detach()})
                
            else:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                if 'mean' in intervention:
                    activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            #clean_logits = model(clean_tokens, attention_mask=attention_mask)

            clean_logits = model.trace({'input_ids': clean_tokens, 'attention_mask':attention_mask}, trace=False)['logits']

            with model.trace({"input_ids": clean_tokens, "attention_mask": attention_mask}):

                node = graph.nodes['input']
                fwd_index = graph.forward_index(node)
                activation_difference[:, :, fwd_index] -= model.transformer.wte.output

                for layer in range(graph.cfg['n_layers']):

                    if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.config.n_head)):
                        #TODO intervene on the input
                        # try to split the input and then pass it to through the entire layer
                        qkv_inp = model.transformer.h[layer].ln_1.input #[20, 20, 768]
                        
                        c_proj_out = [] #[20, 20, 2304]

                        # qkvs = [qkv_inp.clone() for ii in range(12)]

                        # qkvs = [head.unsqueeze(-2) for head in qkvs]
                        # qkvs = torch.cat(qkvs, dim=-2) #[20, 20, 12, 768]

                        # want each QKV for each attention head to be different
                        # input to Q K V, 3: QKV 12: attention head 
                        # update each difference, copy of the input,
                        # what are graph said,

                        qkv_inp = qkv_inp.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 12, 1) # [20, 20, 3, 12, 768]

                        #for ii, letter in enumerate('qkv'):

                        node = graph.nodes[f'a{layer}.h0']
                        prev_index = graph.prev_index(node)
                        bwd_index = graph.backward_index(node, qkv='q', attn_slice=False)

                        bwd_index = slice(bwd_index, bwd_index + 3 * model.config.n_head)

                        if neuron_matrix is not None:
                            update = einsum(activation_difference[:, :, :len(in_graph_matrix[:prev_index, bwd_index])], neuron_matrix[:len(in_graph_matrix[:prev_index, bwd_index])], in_graph_matrix[:prev_index, bwd_index],'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                        else:
                            update = einsum(activation_difference[:, :, :len(in_graph_matrix[:prev_index, bwd_index])], in_graph_matrix[:prev_index, bwd_index],'batch pos previous hidden, previous ... -> batch pos ... hidden')

                        update = update.to(torch.float16) #[20, 20 , 36, 768]

                        update = update.view(update.shape[0], update.shape[1], 3, model.config.n_head, model.config.hidden_size)

                        qkv_inp += update #[20, 20, 3, 12, 768]

                        ln_1_out = model.transformer.h[layer].ln_1(qkv_inp) #[20, 20, 3, 12, 768]

                        attn_weight = model.transformer.h[layer].attn.c_attn.weight.view(model.config.hidden_size, 3, model.config.n_head, model.config.hidden_size // model.config.n_head) #[768, 3, 12, 64]

                        
                        by_head = einsum(ln_1_out, attn_weight, "batch pos qkv head_idx model_dim, model_dim qkv head_idx head_dim -> batch pos qkv head_idx head_dim")

                        by_head = by_head.reshape(by_head.shape[0], by_head.shape[1], -1)

                        by_head += model.transformer.h[layer].attn.c_attn.bias

                        model.transformer.h[layer].attn.c_attn.ouput = by_head

                            # qkv_out = []

                            # for jj in range(12):

                            #     #split_heads = split_heads_nns(qkv_inp, model.config.n_head, model.config.hidden_size) #[20, 20 , 12, 768]

                            #     qkv_in_clone = qkv_inp.clone()

                            #     update_q0 = update[:, :, jj, :] #[20, 20, 768]

                            #     qkv_in_clone += update_q0

                            #     #merged_heads = merge_heads_nns(split_heads, model.config.n_head, model.config.hidden_size)

                            #     qkv_in_clone = model.transformer.h[layer].ln_1(qkv_in_clone) 
                            #     attn_out = model.transformer.h[layer].attn.c_attn(qkv_in_clone) #[20, 20, 2304]

                            #     start = ii * jj * (model.config.hidden_size // model.config.n_head)
                            #     end = start + ii * jj * (model.config.hidden_size // model.config.n_head)

                            #     attn_head_out = attn_out[:, :, start:end] #[20, 20, 64]

                            #     qkv_out.append(attn_head_out)

                            # qkv_out = torch.cat(qkv_out, dim=-1) #[20, 20,768]

                            # c_proj_out.append(qkv_out)

                            # qkv_out = []

                            # qkvs_clone = qkvs.clone()
                            # qkvs_clone += update

                            # qkv_in_clone = model.transformer.h[layer].ln_1(qkvs_clone) # [20, 20, 12, 768]

                            # attn_out = model.transformer.h[layer].attn.c_attn(qkv_in_clone) #[20, 20, 12, 2304]

                            # res = attn_out.split(768, dim=3)

                            # res = res[ii] # [20, 20 , 12, 768]

                            # res_out = torch.zeros(res.shape[0], res.shape[1], 768) 

                            # for jj in range(12):
                            #     start = jj * 64
                            #     end = start + 64
                            #     res_out[start:end] = res[:, :, jj][start:end]

                            #for jj in range(12):

                                #split_heads = split_heads_nns(qkv_inp, model.config.n_head, model.config.hidden_size) #[20, 20 , 12, 768]

                                # qkv_in_clone = qkv_inp.clone()

                                # update_q0 = update[:, :, jj, :] #[20, 20, 768]

                                # qkv_in_clone += update_q0

                                #merged_heads = merge_heads_nns(split_heads, model.config.n_head, model.config.hidden_size)

                                # qkv_in_clone = model.transformer.h[layer].ln_1(qkv_in_clone) 
                                
                                

                                # split by head by QKV
                            #     start = ii * jj * (model.config.hidden_size // model.config.n_head)
                            #     end = start + ii * jj * (model.config.hidden_size // model.config.n_head)

                            #     attn_head_out = attn_out[:, :, start:end] #[20, 20, 64]

                            #     qkv_out.append(attn_head_out)

                            # qkv_out = torch.cat(qkv_out, dim=-1) #[20, 20,768]

                            #c_proj_out.append(res_out) # 3 * [20, 20 , 12, 768]

                    # c_proj_out = torch.cat(c_proj_out, dim=-1)

                    node = graph.nodes[f'a{layer}.h0']
                    fwd_index = graph.forward_index(node)
                    attn_hs = model.transformer.h[layer].attn.c_proj.input

                    attn_hs = attn_hs.view(attn_hs.shape[0], attn_hs.shape[1], model.config.n_head, model.config.hidden_size // model.config.n_head)

                    W_O = model.transformer.h[layer].attn.c_proj.weight.view(model.config.n_head, model.config.hidden_size // model.config.n_head, model.config.hidden_size)

                    by_head = einsum(attn_hs, W_O, "batch pos head_idx head_dim, head_idx head_dim model_dim -> batch pos head_idx model_dim")

                    activation_difference[:, :, fwd_index] -= by_head

                    #### MLP #########
                    if graph.nodes[f'm{layer}'].in_graph:
                        node = graph.nodes[f'm{layer}']
                        prev_index = graph.prev_index(node)
                        bwd_index = graph.backward_index(node)

                        if neuron_matrix is not None:
                            update = einsum(activation_difference[:, :, :len(in_graph_matrix[:prev_index, bwd_index])], neuron_matrix[:len(in_graph_matrix[:prev_index, bwd_index])], in_graph_matrix[:prev_index, bwd_index],'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                        else:
                            update = einsum(activation_difference[:, :, :len(in_graph_matrix[:prev_index, bwd_index])], in_graph_matrix[:prev_index, bwd_index],'batch pos previous hidden, previous ... -> batch pos ... hidden')

                        model.transformer.h[layer].ln_2.input = model.transformer.h[layer].ln_2.input + update

                    node = graph.nodes[f'm{layer}']
                    fwd_index = graph.forward_index(node)
                    activation_difference[:, :, fwd_index] -= model.transformer.h[layer].mlp.output
                            

                #### Logits #########
                node = graph.nodes['logits']
                fwd_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                if neuron_matrix is not None:
                    update = einsum(activation_difference[:, :, :len(in_graph_matrix[:prev_index, bwd_index])], neuron_matrix[:len(in_graph_matrix[:prev_index, bwd_index])], in_graph_matrix[:prev_index, bwd_index],'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                else:
                    update = einsum(activation_difference[:, :, :len(in_graph_matrix[:prev_index, bwd_index])], in_graph_matrix[:prev_index, bwd_index],'batch pos previous hidden, previous ... -> batch pos ... hidden')

                model.transformer.ln_f.input += update

                logits = model.lm_head.output.save()

        torch.save(save_data, f"tensors/corr_tokens_act_diff_nns.pt")

        for i, metric in enumerate(metrics):
            r = metric(logits, clean_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    if len(results) == 1:
        results = results[0]

    return results

def evaluate_baseline_nnsight(model, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], run_corrupted=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluates the model on the given dataloader, without any intervention. This is useful for computing the baseline performance of the model.

    Args:
        model (HookedTransformer): The model to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (List[Callable[[Tensor], Tensor]]): The metrics to evaluate with respect to
        run_corrupted (bool, optional): Whether to evaluate on corrupted examples instead. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of performance scores; if a list, each list entry corresponds to a metric in the input list
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    
    results = [[] for _ in metrics]
    for clean, corrupted, label in tqdm(dataloader):
        clean_tokens, attention_mask, input_lengths, _ = tokenize_plus_nns(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus_nns(model, corrupted)
        with torch.inference_mode():
            corrupted_logits = model.trace({'input_ids': corrupted_tokens, 'attention_mask': attention_mask}, trace=False)["logits"]
            #print(corrupted_logits)
            #print(corrupted_logits.shape)
            logits = model.trace({'input_ids': clean_tokens, 'attention_mask': attention_mask}, trace=False)["logits"]
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_baseline_nns(model, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], run_corrupted=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluates the model on the given dataloader, without any intervention. This is useful for computing the baseline performance of the model.

    Args:
        model (HookedTransformer): The model to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (List[Callable[[Tensor], Tensor]]): The metrics to evaluate with respect to
        run_corrupted (bool, optional): Whether to evaluate on corrupted examples instead. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of performance scores; if a list, each list entry corresponds to a metric in the input list
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    
    results = [[] for _ in metrics]
    for clean, corrupted, label in tqdm(dataloader):
        clean_tokens, attention_mask, input_lengths, _ = tokenize_plus_nns(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus_nns(model, corrupted)
        with torch.inference_mode():
            corrupted_logits = model.trace({'input_ids': corrupted_tokens, 'attention_mask': attention_mask}, trace=False)
            corrupted_logits = corrupted_logits["logits"]
            #print(corrupted_logits)
            #print(corrupted_logits.shape)
            logits = model.trace({'input_ids': clean_tokens, 'attention_mask': attention_mask}, trace=False)["logits"]
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results


def split_heads_nns(hidden_states, num_heads, hidden_size):
    split_heads = []

    for head in range(num_heads):

        hs_clone = hidden_states.clone()

        start = head * (hidden_size // num_heads)
        end = start + (hidden_size // num_heads)

        hs_clone[:start] = 0
        hs_clone[end:] = 0

        split_heads.append(hs_clone)

    split_heads = [head.unsqueeze(-2) for head in split_heads]
    split_heads = torch.cat(split_heads, dim=-2)

    return split_heads