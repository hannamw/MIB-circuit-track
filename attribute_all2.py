from functools import partial

import torch
import argparse
import os
import pickle
#from nnsight.models import UnifiedTransformer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm 

from graph import Graph
from circuit_loading import load_graph_from_json, load_graph_from_pt

from dataset import EAPDataset, HFEAPDataset
from attribute import attribute
from attribute_node import attribute_node
from metrics import get_metric
from transformers import BitsAndBytesConfig
from evaluate_graph import (
    evaluate_graph, evaluate_baseline, evaluate_area_under_curve, area_under_roc
)
from huggingface_hub import hf_hub_download

def load_interpbench_model():
    hf_cfg = hf_hub_download("cybershiptrooper/InterpBench", subfolder="ioi", filename="ll_model_cfg.pkl")
    # hf_model = hf_hub_download("cybershiptrooper/InterpBench", subfolder=task_name, filename="ll_model.pth")
    it_model_path = "interpbench/ioi_all_splits/ll_model_100_100_80.pth"

    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        # Some cases in InterpBench have the config as a HookedTransformerConfig object instead of a dict
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = "cuda"

    # Small hack to enable evaluation mode in the IOI model, that has a different config during training
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(it_model_path, map_location="cuda"))
    return model


def run_circuit_discovery(task, model_name, ablation="patching", method="EAP-IG-inputs", nodes=False, transformers_cache_dir=None):
    outdir = f"circuits/{task}_{model_name}_{method}_{ablation}"
    if nodes:
        outdir += "_node"
    if os.path.exists(os.path.join(outdir, "train.json")):
        return

    if os.path.exists(f"ablations/{model_name}/{task}_oa.pkl"):
        ablation_path = f"ablations/{model_name}/{task}_oa.pkl"
    else:
        ablation_path = None

    # load model
    if model_name == "gpt2":
        model = HookedTransformer.from_pretrained("gpt2-small")
    elif model_name == "qwen2.5":
        model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager", torch_dtype=torch.bfloat16)
    elif model_name == "llama3":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", attn_implementation="eager",
                                                  quantization_config=quantization_config, torch_dtype=torch.bfloat16, cache_dir=transformers_cache_dir)
    elif model_name == "gemma2":
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = load_interpbench_model()
        if task != "ioi":
            raise ValueError("InterpBench is only compatible with the IOI task!")
        
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True if model_name in ("gemma2", "qwen2.5", "llama3") else False

    if task.endswith(("addition", "subtraction", "multiplication", "division")):
        operator = task.split("_")[-1]
    else:
        operator = None

    # load data
    dataset_name = "mech-interp-bench/copycolors_mcqa" if task == "mcqa" else "mech-interp-bench/ioi" if task == "ioi" \
                   else "mech-interp-bench/arithmetic_addition" if task == "arithmetic_addition" \
                   else "mech-interp-bench/arithmetic_subtraction" if task == "arithmetic_subtraction" \
                   else "mech-interp-bench/arithmetic_multiplication" if task == "arithmetic_multiplication" \
                   else "mech-interp-bench/arithmetic_division" if task == "arithmetic_division" \
                   else "mech-interp-bench/arc_easy" if task == "arc_easy" else "mech-interp-bench/arc_challenge" if task == "arc_challenge" \
                   else None

    if "addition" in task or "subtraction" in task:
        task = "arithmetic"
    if "_easy" in task or "_challenge" in task:
        task = "arc"

    if task == "ewok":
        dataset_name = "ewok-core/ewok-core-1.0"
    num_examples = 1000 if task == "ioi" else 100
    example_domain = "+" if "addition" in dataset_name else "-" if "subtraction" in dataset_name \
                     else "*" if "multiplication" in dataset_name else "/"
    dataset = HFEAPDataset(dataset_name, model.tokenizer, task=task, num_examples=num_examples, example_domain=example_domain)
    # dataset_control = HFEAPDataset(dataset_name, model.tokenizer, task=task, num_examples=num_examples, control=True,)
    batch_size = 1 if model_name == "llama3" else 2 if task == "arc" else 10
    dataloader = dataset.to_dataloader(batch_size)
    # dataloader_control = dataset_control.to_dataloader(batch_size)
    if task == "ewok":
        metric_fn = get_metric("sequence_logprob", task, model.tokenizer, model)
    else:
        metric_fn = get_metric("logit_diff", task, model.tokenizer, model)


    # run circuit discovery
    g = Graph.from_model(model, neuron_level=nodes)
    # g_control = Graph.from_model(model)

    if nodes:
        attribute_node(model, g, dataloader, partial(metric_fn, mean=True, loss=True), method, intervention=ablation,
              intervention_dataloader=dataloader, ig_steps=5, neuron=True)
    else:
        attribute(model, g, dataloader, partial(metric_fn, mean=True, loss=True), method, intervention=ablation,
                intervention_dataloader=dataloader, ig_steps=5)
    # attribute(model, g_control, dataloader_control, partial(metric_fn, mean=True, loss=True), method,
    #           intervention=ablation, intervention_dataloader=dataloader_control, ig_steps=5)
    
    # save circuits
    os.makedirs(outdir, exist_ok=True)
    train_outfile = os.path.join(outdir, "train.json")
    outfile_control = os.path.join(outdir, "control.json")
    if nodes:
        g.to_pt(train_outfile)
    else:
        g.to_json(train_outfile)
    # g_control.to_json(outfile_control)


def run_evaluation(task, model_name, split="train", ablation="patching", method="EAP-IG-activations", use_accuracy=False,
                   run_greedy=False, eval_random=False, nodes=False, transformers_cache_dir=None):
    circuit_dir = f"circuits/{task}_{model_name}_{method}_{ablation}"
    if method == "ugs":
        m = "gpt2-small" if model_name == "gpt2" else "qwen" if model_name == "qwen2.5" else "?"
        circuit_dir = f"circuits/ugs/{task}/{m}/0.001/"
    if nodes:
        circuit_dir += "_node"
    if os.path.exists(os.path.join(circuit_dir, "results", f"eval_{split}_patch_abs.pkl")) or \
        os.path.exists(os.path.join(circuit_dir, "results", f"eval_{split}_patch.pkl")):
        return

    # load model
    if model_name == "gpt2":
        model = HookedTransformer.from_pretrained("gpt2-small")
    elif model_name == "qwen2.5":
        model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager", torch_dtype=torch.bfloat16)
    elif model_name == "llama3":
        model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", attn_implementation="eager", torch_dtype=torch.bfloat16, cache_dir=transformers_cache_dir)
    elif model_name == "gemma2":
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = load_interpbench_model()
        if task != "ioi":
            raise ValueError("InterpBench is only compatible with the IOI task!")
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True if model_name in ("gemma2", "qwen2.5", "llama3") else False

    if task.endswith(("addition", "subtraction", "multiplication", "division")):
        operator = task.split("_")[-1]
    else:
        operator = None

    # load data
    dataset_name = "mech-interp-bench/copycolors_mcqa" if task == "mcqa" else "mech-interp-bench/ioi" if task == "ioi" \
                   else "mech-interp-bench/arithmetic_addition" if task == "arithmetic_addition" \
                   else "mech-interp-bench/arithmetic_subtraction" if task == "arithmetic_subtraction" \
                   else "mech-interp-bench/arithmetic_multiplication" if task == "arithmetic_multiplication" \
                   else "mech-interp-bench/arithmetic_division" if task == "arithmetic_division" \
                   else "mech-interp-bench/arc_easy" if task == "arc_easy" else "mech-interp-bench/arc_challenge" if task == "arc_challenge" \
                   else None

    if "addition" in task or "subtraction" in task:
        task = "arithmetic"
    if "_easy" in task or "_challenge" in task:
        task = "arc"

    example_domain = "+" if "addition" in dataset_name else "-" if "subtraction" in dataset_name \
                     else "*" if "multiplication" in dataset_name else "/"
    num_examples = 1000 if task == "ioi" else 100
    if task == "mcqa" and split in ("validation", "test"):
        num_examples = 50
    dataset = HFEAPDataset(dataset_name, model.tokenizer, task=task, num_examples=num_examples, split=split, example_domain=example_domain)
    # dataset_control = HFEAPDataset(dataset_name, model.tokenizer, task=task, num_examples=num_examples, split=split, control=True)
    batch_size = 1 if model_name == "llama3" else 2 if task == "arc" else 20 if model_name == "gpt2" else 10
    dataloader = dataset.to_dataloader(batch_size)
    # dataloader_control = dataset_control.to_dataloader(batch_size)
    if use_accuracy:
        metric_fn = get_metric("acc", task, model.tokenizer, model)
    else:
        metric_fn = get_metric("logit_diff", task, model.tokenizer, model)

    if eval_random:
        for seed in (1, 2, 3):
            circuit_dir = f"circuits/random{seed}_{model_name}/"
            train_circuitpath = os.path.join(circuit_dir, "train.json")
            graph = load_graph_from_json(train_circuitpath)
            if os.path.exists(os.path.join(circuit_dir, f"results_{task}/", f"eval_{split}_patch.pkl")):
                continue
            if "interpbench" in model_name:
                reference_graph = load_graph_from_json("interpbench/interpbench_graph.json")
                results_patch = area_under_roc(reference_graph, graph)
            else:
                results_patch = evaluate_area_under_curve(model, graph, dataloader, partial(metric_fn, loss=False, mean=False),
                                        intervention='patching', absolute=False, no_normalize=use_accuracy,
                                        apply_greedy=run_greedy)
            outname = f"eval_{split}_patch_"
            outname += "acc_" if use_accuracy else ""
            outname += "greedy_" if run_greedy else ""
            outname = outname[:-1] + ".pkl"
            outdir = os.path.join(circuit_dir, f"results_{task}/")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, outname), "wb") as handle:
                pickle.dump(results_patch, handle)
        return
    # control_circuitpath = os.path.join(circuit_dir, "control.json")

    train_circuitpath = os.path.join(circuit_dir, "train.json")
    if method == "ugs":
        # override path; different directory structure for UGS
        train_circuitpath = os.path.join(circuit_dir, "graph.json")
    if nodes:
        graph = load_graph_from_pt(train_circuitpath)
    else:
        graph = load_graph_from_json(train_circuitpath)
    if model_name == "interpbench":
        reference_graph = load_graph_from_json("interpbench/interpbench_graph.json")
    level = 'neuron' if nodes else 'edge'
    # graph_control = load_graph_from_json(control_circuitpath)

    # results_mean_abs = evaluate_area_under_curve(model, graph, dataloader, partial(metric_fn, loss=False, mean=False),
    #                                 intervention='mean', intervention_dataloader=dataloader, absolute=True)
    # results_mean_control_abs = evaluate_area_under_curve(model, graph_control, dataloader_control, partial(metric_fn, loss=False, mean=False),
    #                                 intervention='mean', intervention_dataloader=dataloader_control, absolute=True)
    # results_mean = evaluate_area_under_curve(model, graph, dataloader, partial(metric_fn, loss=False, mean=False),
    #                                 intervention='mean', intervention_dataloader=dataloader, absolute=False)
    # results_mean_control = evaluate_area_under_curve(model, graph_control, dataloader_control, partial(metric_fn, loss=False, mean=False),
    #                                 intervention='mean', intervention_dataloader=dataloader_control, absolute=False)

    if not os.path.exists(os.path.join(circuit_dir, "results", f"eval_{split}_patch_abs.pkl")):
        if model_name == "interpbench":
            results_patch_abs = None
        else:
            results_patch_abs = evaluate_area_under_curve(model, graph, dataloader, partial(metric_fn, loss=False, mean=False),
                                        intervention='patching', absolute=True, no_normalize=use_accuracy,
                                        apply_greedy=run_greedy, level=level)
    else:
        results_patch_abs = None
    # results_patch_control_abs = evaluate_area_under_curve(model, graph_control, dataloader_control, partial(metric_fn, loss=False, mean=False),
    #                                 intervention='patching', absolute=True, no_normalize=use_accuracy)
    if not os.path.exists(os.path.join(circuit_dir, "results", f"eval_{split}_patch.pkl")):
        if model_name == "interpbench":
            results_patch = area_under_roc(reference_graph, graph)
        else:
            results_patch = evaluate_area_under_curve(model, graph, dataloader, partial(metric_fn, loss=False, mean=False),
                                        intervention='patching', absolute=False, no_normalize=use_accuracy,
                                        apply_greedy=run_greedy, level=level)
    else:
        results_patch = None
    # results_patch_control = evaluate_area_under_curve(model, graph_control, dataloader_control, partial(metric_fn, loss=False, mean=False),
    #                                 intervention='patching', absolute=False, no_normalize=use_accuracy)
    results_objs = (results_patch_abs, results_patch)
                    # results_mean_abs, results_mean_control_abs, results_mean, results_mean_control,
                   # results_patch_abs, results_patch_control_abs, results_patch, results_patch_control)
    for i, obj in enumerate(results_objs):
        if obj is None:
            continue
        outname = f"eval_{split}_"
        # outname += "control_" if i % 2 == 1 else ""
        outname += "patch_" if i < 4 else "patch_"
        outname += "abs_" if i % 2 == 0 else ""
        outname += "acc_" if use_accuracy else ""
        outname += "greedy_" if run_greedy else ""
        outname = outname[:-1] + ".pkl"
        outdir = os.path.join(circuit_dir, "results/")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, outname), "wb") as handle:
            pickle.dump(obj, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--transformers_cache_dir", type=str, default=None)
    args = parser.parse_args()

    RUN_ID_TO_METHODS = {0: ("EAP", "EAP-IG-inputs"),
                        1: ("EAP", "EAP-IG-inputs", "information-flow-routes"),
                        2: ("EAP-IG-activations",),
                        3: ("EAP-IG-activations",)
    }

    RUN_ID_TO_TASKS = {0: ("arc_easy", "arc_challenge"),
                       1: ("arc_challenge",),
                       2: ("arc_easy",),
                       3: ("arc_challenge",)}
    
    MODELS = ("llama3",)
    METHODS = RUN_ID_TO_METHODS[args.run_id]
    TASKS = RUN_ID_TO_TASKS[args.run_id]
    ABLATIONS = ("patching", "mean") if args.run_id in (0, 1) else ("patching",)
    node_patching = (args.run_id == 0)
    
    def _run_discovery_and_eval(task, model, ablation, method, nodes=False):
        run_circuit_discovery(task, model, ablation=ablation, method=method,
                              nodes=nodes, transformers_cache_dir=args.transformers_cache_dir)
        for split in ("train", "validation", "test"):
            print(f"evaluating at split {split}")
            if method == "information-flow-routes":
                run_greedy = True
            else:
                run_greedy = False
            run_evaluation(task, model, ablation=ablation, method=method, split=split,
                            use_accuracy=False, run_greedy=run_greedy, eval_random=False,
                            nodes=nodes, transformers_cache_dir=args.transformers_cache_dir)

    for model in MODELS:
        for task in TASKS:
            if task == "mcqa" and model == "gpt2":
                continue
            if task == "arc" and model == "llama3":
                continue
            if task == "arithmetic" and model in ("gemma2", "qwen2.5", "gpt2"):
                continue
            for ablation in ABLATIONS:
                for method in METHODS:
                    if method in ("EAP-IG-inputs", "information-flow-routes") and ablation != "patching":
                        continue
                    print(f"Running {method} ({ablation}) on {task} using {model}...")
                    _run_discovery_and_eval(task, model, ablation, method, nodes=node_patching)
                    print()