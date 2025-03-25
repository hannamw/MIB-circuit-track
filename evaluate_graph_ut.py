from typing import Callable, List, Union, Literal, Optional
import math
from functools import partial

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from tqdm import tqdm
from einops import einsum, rearrange

from attribute import tokenize_plus, compute_mean_activations, load_ablations
from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode
from circuit_loading import load_graph_from_json, load_graph_from_pt
from dataset import HFEAPDataset

import unittest

def evaluate_graph_residual(model: UnifiedTransformer, graph: Graph, dataloader: DataLoader,
                  metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]],
                  quiet=False, intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching',
                  intervention_dataloader: Optional[DataLoader]=None) -> Union[Tensor, List[Tensor]]:
    """Evaluate a circuit using nnsight tracing for interventions."""
    # Ensure model is configured correctly
    assert model.cfg.use_attn_result, "Model must use attention result"
    if model.cfg.n_key_value_heads:
        assert model.cfg.ungroup_grouped_query_attention, "Model must ungroup grouped queries"

    # Precompute means if needed
    if 'mean' in intervention:
        assert intervention_dataloader, "Intervention dataloader required for mean ablation"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position)
        means = means.unsqueeze(0)

    graph.prune()
    device = model.cfg.device
    dtype = model.cfg.dtype

    # Prepare masks
    in_graph_matrix = (1 - graph.in_graph.to(device=device, dtype=dtype))
    neuron_matrix = (1 - graph.neurons_in_graph.to(device, dtype)) if graph.neurons_in_graph else None

    # Convert metrics to list
    metrics = [metrics] if not isinstance(metrics, list) else metrics
    results = [[] for _ in metrics]

    for batch in tqdm(dataloader, disable=quiet):
        clean, corrupted, labels = batch
        clean_tokens = model.tokenizer(clean, return_tensors='pt', padding=True).to(device)
        corrupted_tokens = model.tokenizer(corrupted, return_tensors='pt', padding=True).to(device)
        with model.trace(clean_tokens) as clean_invoke:
            # Capture ALL residual stream points
            clean_residuals = {
                f'blocks.{layer}.hook_resid_pre': model.blocks[layer].hook_resid_pre.input[0][0].save()
                for layer in range(model.cfg.n_layers)
            }
            clean_logits = model.output.save()

        # Corrupted pass
        with model.trace(corrupted_tokens) as corrupted_invoke:
            corrupted_residuals = {
                f'blocks.{layer}.hook_resid_pre': model.blocks[layer].hook_resid_pre.input[0][0].save()
                for layer in range(model.cfg.n_layers)
            }
            corrupted_logits = model.output.save()

        # Compute residual differences
        residual_diffs = {
            k: corrupted_residuals[k].value - clean_residuals[k].value
            for k in clean_residuals.keys()
        }
        
        # Single intervention pass
        with model.trace(clean_tokens):
            # Vectorized residual stream modification
            for layer in range(model.cfg.n_layers):
                # Get mask for this layer's residual connections
                layer_mask = graph.get_layer_mask(layer).to(model.cfg.device)  # Implement this in Graph class
                print(layer_mask)
                
                # Get original residual stream
                resid = model.blocks[layer].hook_resid_pre.input[0][0]
                
                # Apply modifications
                modified_resid = resid + residual_diffs[f'blocks.{layer}.hook_resid_pre'] * layer_mask
                model.blocks[layer].hook_resid_pre.input[0][0][:] = modified_resid

            # Final logits comparison
            logits = model.output.save()
        
        # Compute metrics
        for i, metric in enumerate(metrics):
            results[i].append(metric(logits.value, clean_logits.value, labels))
    
    return [torch.cat(r) for r in results] if len(results) > 1 else torch.cat(results[0])


def evaluate_graph(model: UnifiedTransformer, graph: Graph, dataloader: DataLoader,
                  metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]],
                  quiet=False, intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching',
                  intervention_dataloader: Optional[DataLoader]=None) -> Union[Tensor, List[Tensor]]:
    
    """Evaluate a circuit using nnsight tracing for interventions."""
    # Ensure model is configured correctly
    assert model.cfg.use_attn_result, "Model must use attention result"
    if model.cfg.n_key_value_heads:
        assert model.cfg.ungroup_grouped_query_attention, "Model must ungroup grouped queries"

    # Precompute means if needed
    if 'mean' in intervention:
        assert intervention_dataloader, "Intervention dataloader required for mean ablation"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position)
        means = means.unsqueeze(0)

    graph.prune()
    device = model.cfg.device
    dtype = model.cfg.dtype

    # Prepare masks
    in_graph_matrix = (1 - graph.in_graph.to(device=device, dtype=dtype))
    neuron_matrix = (1 - graph.neurons_in_graph.to(device, dtype)) if graph.neurons_in_graph else None

    # Convert metrics to list
    metrics = [metrics] if not isinstance(metrics, list) else metrics
    results = [[] for _ in metrics]

    for batch in tqdm(dataloader, disable=quiet):
        clean, corrupted, labels = batch
        clean_tokens = model.tokenizer(clean, return_tensors='pt', padding=True).to(device)
        corrupted_tokens = model.tokenizer(corrupted, return_tensors='pt', padding=True).to(device)

        # Phase 1: Capture corrupted activations
        corrupted_activations = {}
        with model.trace(corrupted_tokens) as tracer:
            # Populate all required activations
            for node in graph.nodes.values():
                if isinstance(node, AttentionNode):
                    for letter in 'qkv':
                        corrupted_activations[f"blocks.{node.layer}.hook_{letter}_input"] = \
                            getattr(model.blocks[node.layer].attn, f'hook_{letter}').input[0][0].save()
                elif isinstance(node, MLPNode):
                    corrupted_activations[f"blocks.{node.layer}.hook_mlp_in"] = model.blocks[node.layer].mlp.input[0][0].save()
        
        # Phase 2: Capture clean activations
        clean_activations = {}
        with model.trace(clean_tokens) as tracer:
            for node in graph.nodes.values():
                if isinstance(node, AttentionNode):
                    for letter in 'qkv':
                        clean_activations[f"blocks.{node.layer}.hook_{letter}_input"] = \
                            getattr(model.blocks[node.layer].attn, f'hook_{letter}').input[0][0].save()
                elif isinstance(node, MLPNode):
                    clean_activations[f"blocks.{node.layer}.hook_mlp_in"] = model.blocks[node.layer].mlp.input[0][0].save()
            clean_logits = model.output.save()

        # Compute activation differences
        activation_diff = {
            k: corrupted_activations[k].value - clean_activations[k].value
            for k in corrupted_activations.keys()
        }

        # Phase 3: Apply interventions using module inputs
        with model.trace(clean_tokens, scan=False) as tracer:
            # Process all edges in the graph
            for edge in graph.edges.values():
                if edge.in_graph:
                    continue  # Skip edges included in the circuit

                # Get the target module and hook type
                if isinstance(edge.child, AttentionNode):
                    module = model.blocks[edge.child.layer].attn
                    input_attr = f"hook_{edge.qkv}"
                    orig = getattr(module, input_attr).input[0][0]
                elif isinstance(edge.child, MLPNode):
                    module = model.blocks[edge.child.layer].mlp
                    orig = module.input[0][0]
                else:
                    continue
                
                # Compute activation difference for this hook
                delta = activation_diff[edge.hook]

                # Apply neuron-level masking
                if neuron_matrix is not None:
                    src_idx = graph.forward_index(edge.parent)
                    mask = neuron_matrix[src_idx].reshape_as(orig)
                    delta *= mask

                # Apply edge mask and modify input
                orig = orig + delta * in_graph_matrix[edge.matrix_index].to(delta.dtype)

            # Handle logits output separately
            if not graph.nodes['logits'].in_graph:
                output = model.output.value
                modified_output = output + activation_diff['logits']
                model.output = modified_output

            logits = model.output.save()

        # Compute metrics
        for i, metric in enumerate(metrics):
            results[i].append(metric(logits.value, clean_logits.value, labels))

    return [torch.cat(r) for r in results] if len(results) > 1 else torch.cat(results[0])


def evaluate_baseline(model: UnifiedTransformer, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], run_corrupted=False) -> Union[torch.Tensor, List[torch.Tensor]]:
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
        clean_tokens, attention_mask, input_lengths, _ = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            logits = model(clean_tokens, attention_mask=attention_mask)
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


def evaluate_area_under_curve(model: UnifiedTransformer, graph: Graph, dataloader, metrics, quiet:bool=False, 
                              level:Literal['edge', 'node','neuron']='edge', log_scale:bool=True, absolute:bool=True,
                              intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching',
                              intervention_dataloader:DataLoader=None, ablation_path:str=None,
                              no_normalize:Optional[bool]=False, apply_greedy:bool=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention, intervention_dataloader=intervention_dataloader).mean().item()
    
    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)
    
    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    weighted_edge_counts = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct*100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            assert level == 'edge', "Greedy application only supported for edge-level evaluation"
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)
        
        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader, ablation_path=ablation_path).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)
    
    area_under = 0.
    area_from_100 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (percentages[i_2] - percentages[i_1]) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_100 += trapezoidal 
        
        trapezoidal = (percentages[i_2] - percentages[i_1]) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, area_under, area_from_100, average, faithfulnesses


def compare_graphs(reference: Graph, hypothesis: Graph, by_node: bool = False):
    # Track {true, false} {positives, negatives}
    TP, FP, TN, FN = 0, 0, 0, 0
    total = 0

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges

    for obj in ref_objs.values():
        total += 1
        if obj.name not in hyp_objs:
            if obj.in_graph:
                TP += 1
            else:
                FP += 1
            continue
            
        if obj.in_graph and hyp_objs[obj.name].in_graph:
            TP += 1
        elif obj.in_graph and not hyp_objs[obj.name].in_graph:
            FN += 1
        elif not obj.in_graph and hyp_objs[obj.name].in_graph:
            FP += 1
        elif not obj.in_graph and not hyp_objs[obj.name].in_graph:
            TN += 1
    
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # f1 = (2 * precision * recall) / (precision + recall)
    TP_rate = recall
    FP_rate = FP / (FP + TN)

    return {"precision": precision,
            "recall": recall,
            "TP_rate": TP_rate,
            "FP_rate": FP_rate}

def area_under_roc(reference: Graph, hypothesis: Graph, by_node: bool = False):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges
    
    num_objs = len(ref_objs.values())
    for pct in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1):
        this_num_objs = pct * num_objs
        if by_node:
            raise NotImplementedError("")
        else:
            hypothesis.apply_greedy(this_num_objs)
        scores = compare_graphs(reference, hypothesis)
        tpr_list.append(scores["TP_rate"])
        fpr_list.append(scores["FP_rate"])
        precision_list.append(scores["precision"])
        recall_list.append(scores["recall"])
    
    return {"TPR": tpr_list, "FPR": fpr_list,
            "precision": precision_list, "recall": recall_list}


class EquivalenceTests(unittest.TestCase):
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name

    def setUpClass(self, graph_path=None):
        # Original TransformerLens model
        from transformer_lens import HookedTransformer
        self.tl_model = HookedTransformer.from_pretrained(self.model_name)
        self.tl_model.cfg.use_attn_result = True
        
        # New nnsight model
        from nnsight.models.UnifiedTransformer import UnifiedTransformer
        self.ns_model = UnifiedTransformer(self.model_name)
        self.ns_model.cfg.use_attn_result = True
        
        # Sample inputs
        self.dataset = HFEAPDataset("mech-interp-bench/ioi", self.ns_model.tokenizer,
                                    task="ioi", num_examples=1, split="train")

        if graph_path is None:
            graph_path = "circuits/ioi_gpt2_EAP_patching/train.json"
        self.graph = load_graph_from_json(graph_path)
        
        # Sample circuit
        # self.graph = Graph()
        # self.graph.add_node(AttentionNode(layer=0, head=0))
        # self.graph.add_node(MLPNode(layer=1))
        # self.graph.add_node(LogitNode())

    def test_mean_ablation_equivalence(self):
        """Compare mean ablation results"""
        # Create small dataset for mean computation
        mean_dataset = [("Sample text", "Corrupted", 0) for _ in range(5)]
        
        # Original implementation
        from evaluate_graph import evaluate_graph as tl_evaluate
        tl_result = tl_evaluate(
            self.tl_model,
            self.graph,
            [(self.clean_text, self.corrupted_text, 0)],
            lambda logits, *_: logits[0,-1].sum(),
            intervention='mean',
            intervention_dataloader=mean_dataset
        )
        
        # New implementation
        ns_result = evaluate_graph(
            self.ns_model,
            self.graph,
            [(self.clean_text, self.corrupted_text, 0)],
            lambda logits, *_: logits[0,-1].sum(),
            intervention='mean',
            intervention_dataloader=mean_dataset
        )
        
        print("TransformerLens:", tl_result)
        print("==========")
        print("NNsight:", ns_result)
    
    def test_neuron_masking_equivalence(self, intervention='patching', seed=12):
        """Compare neuron-level masking results"""
        # Create neuron mask
        torch.manual_seed(seed)
        dataloader = self.dataset.to_dataloader(batch_size=1)
        self.graph.apply_topn(512, absolute=True)

        # neuron_mask = torch.rand(self.tl_model.cfg.d_model) > 0.5
        # self.graph.set_neuron_mask(0, self.tl_model.cfg.d_model, neuron_mask)
        
        # Original implementation
        from evaluate_graph import evaluate_graph as tl_evaluate
        tl_result = tl_evaluate(
            self.tl_model,
            self.graph,
            dataloader,
            lambda logits, *_: logits[0,-1],
            intervention=intervention
        )
        
        # New implementation
        ns_result = evaluate_graph_residual(
            self.ns_model,
            self.graph,
            dataloader,
            lambda logits, *_: logits[0,-1],
            intervention=intervention
        )

        print("TransformerLens:", tl_result)
        print("==========")
        print("NNsight:", ns_result)

if __name__ == "__main__":
    tests = EquivalenceTests("gpt2")
    tests.setUpClass()
    tests.test_neuron_masking_equivalence()