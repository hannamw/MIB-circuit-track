from typing import List, Dict, Union, Tuple, Literal, Optional, Set
from collections import defaultdict
from pathlib import Path 
import json
import heapq

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np
import pygraphviz as pgv

from visualization import EDGE_TYPE_COLORS, generate_random_color

class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs, 
    while the out_hook gets its outputs.
    """
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    score: Optional[float]
    qkv_inputs: Optional[List[str]]
    neurons: Optional[torch.Tensor]
    neuron_scores: Optional[torch.Tensor]

    def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple,
                 score: Optional[float]=None, qkv_inputs: Optional[List[str]]=None, neurons: Optional[torch.Tensor]=None, neuron_scores: Optional[torch.Tensor]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook 
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.score = score
        self.qkv_inputs = qkv_inputs
        self.neurons = neurons
        self.neuron_scores = neuron_scores

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Node({self.name}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class LogitNode(Node):
    def __init__(self, n_layers:int):
        name = 'logits' 
        index = slice(None) 
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index)
        
class MLPNode(Node):
    def __init__(self, layer: int, neurons: Optional[torch.Tensor] = None, neuron_scores: Optional[torch.Tensor] = None):
        name = f'm{layer}'
        index = slice(None)
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index,
                         neurons=neurons, neuron_scores=neuron_scores)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int, neurons: Optional[torch.Tensor] = None):
        name = f'a{layer}.h{head}' 
        self.head = head
        index = (slice(None), slice(None), head) 
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, qkv_inputs=[f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'], neurons=neurons)

class InputNode(Node):
    def __init__(self, neurons: Optional[torch.Tensor] = None):
        name = 'input' 
        index = slice(None) 
        super().__init__(name, 0, '', "hook_embed", index, neurons=neurons)  #"blocks.0.hook_resid_pre", index)

class Edge:
    graph: 'Graph'
    name: str
    parent: Node 
    child: Node 
    hook: str
    index: Tuple
    def __init__(self, graph: 'Graph', parent: Node, child: Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        self.graph = graph
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent 
        self.child = child
        self.qkv = qkv
        self.score = 0
        self.in_graph = True
        
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook
            
            
    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF0000"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    n_forward: int 
    n_backward: int
    cfg: HookedTransformerConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        edge = Edge(self, parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)
        
        
    def prev_index(self, node: Node) -> Union[int, slice]:
        """Return the forward index before which all nodes contribute to the input of the given node

        Args:
            node (Node): The node to get the prev forward index of

        Returns:
            Union[int, slice]: an index representing the prev forward index of the node
        """
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
            # raise ValueError(f"No forward for logits node")
        elif isinstance(node, MLPNode):
            if self.cfg['parallel_attn_mlp']:
                return 1 + node.layer * (self.cfg['n_heads'] + 1)
            else:
                return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def forward_index(self, node:Node, attn_slice=True):
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
            # raise ValueError(f"No forward for logits node")
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")        

    def backward_index(self, node:Node, qkv=None, attn_slice=True):
        if isinstance(node, InputNode):
            raise ValueError(f"No backward for input node")
        elif isinstance(node, LogitNode):
            return -1
        elif isinstance(node, MLPNode):
            return (node.layer) * (3 * self.cfg['n_heads'] + 1) + 3 * self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            i = node.layer * (3 * self.cfg['n_heads'] + 1) + ('qkv'.index(qkv) * self.cfg['n_heads'])
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
        
    def get_dst_nodes(self):
        heads = []
        for layer in range(self.cfg['n_layers']):
            for letter in 'qkv':
                for attention_head in range(self.cfg['n_heads']):
                    heads.append(f'a{layer}.h{attention_head}<{letter}>')
            heads.append(f'm{layer}')
        heads.append('logits')
        return heads

    def scores(self, nonzero=False, in_graph=False, sort=True):
        s = torch.tensor([edge.score for edge in self.edges.values() if edge.score != 0 and (edge.in_graph or not in_graph)]) if nonzero else torch.tensor([edge.score for edge in self.edges.values()])
        return torch.sort(s).values if sort else s
    
    def weighted_edge_count(self) -> float:
        """Generates a count of the edges, weighted by number of neurons included if applicable

        Returns:
            float: weighted edge count
        """
        weighted_count = 0
        for edge in self.edges.values():
            if edge.in_graph:
                if edge.parent.neurons is not None:
                    weighted_count += edge.parent.neurons.sum() / edge.parent.neurons.size(0)
                else:
                    weighted_count += 1
        return weighted_count
    
    def weighted_node_count(self) -> float:
        """Generates a count of the nodes, weighted by number of neurons included and percentage of
           possible out-edges from this node if applicable
        
        Returns:
            float: weighted node count
        """
        weighted_count = 0
        for node in self.nodes.values():
            if node.in_graph:
                edge_pct = 0
                for edge in self.edges.values():
                    if edge.parent == node:
                        if edge.in_graph:
                            edge_pct += 1
                        total += 1
                edge_pct /= total
                if node.neurons is not None:
                    weighted_count += edge_pct * (node.neurons.sum() / node.neurons.size(0))
                else:
                    weighted_count += edge_pct

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())
    
    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())

    def apply_threshold(self, threshold: float, absolute: bool):
        # include all edges with a score above the the given threshold
        threshold = float(threshold)
        for node in self.nodes.values():
            node.in_graph = True 
            
        for edge in self.edges.values():
            edge.in_graph = abs(edge.score) >= threshold if absolute else edge.score >= threshold
    
    def apply_topn(self, n:int, absolute: bool, node=False, neuron=False):
        def abs_id(s: float):
            return abs(s) if absolute else s
        
        # get top-n nodes
        if node:
            if neuron:
                non_logit_nodes = [node for node in self.nodes.values() if not isinstance(node, LogitNode)]
                neuron_scores = torch.cat([node.neuron_scores for node in self.nodes.values() if not isinstance(node, LogitNode)])
                top_n_score = neuron_scores.sort(descending=True)[n]
                
                self.nodes['logits'].in_graph = True
                for i, node in enumerate(non_logit_nodes):
                    node.neurons = node.neuron_scores >= top_n_score
                    node.in_graph = torch.any(node.neurons)
                    
                for edge in self.edges.values():
                    edge.in_graph = edge.parent.in_graph and edge.child.in_graph
                
            else:
                assert all(node.score is not None for node in self.nodes.values() if not isinstance(node, LogitNode)), "All non-logit nodes must have a score to apply top-n nodes"
                sorted_nodes = sorted([node for node in self.nodes.values() if not isinstance(node, LogitNode)], key = lambda node: abs_id(node.score), reverse=True)
                
                self.nodes['logits'].in_graph = True
                for node in sorted_nodes[:n]:
                    node.in_graph = True
                for node in sorted_nodes[n:]:
                    node.in_graph = False
                    
                for edge in self.edges.values():
                    edge.in_graph = edge.parent.in_graph and edge.child.in_graph

        # get top-n edges
        else:
            if neuron:
                raise ValueError("Neuron and edge-level top-n not supported; choose one or the other, or provide the circuit yourself")
            for node in self.nodes.values():
                node.in_graph = False
            
            sorted_edges = sorted(list(self.edges.values()), key = lambda edge: abs_id(edge.score), reverse=True)
            for edge in sorted_edges[:n]:
                edge.in_graph = True 
                edge.parent.in_graph = True 
                edge.child.in_graph = True 

            for edge in sorted_edges[n:]:
                edge.in_graph = False

    def apply_greedy(self, n_edges, reset=True, absolute: bool=True):
        if reset:
            for node in self.nodes.values():
                node.in_graph = False 
            for edge in self.edges.values():
                edge.in_graph = False
            self.nodes['logits'].in_graph = True

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph], key = lambda edge: abs_id(edge.score), reverse=True)

        edges = heapq.merge(candidate_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key = lambda edge: abs_id(edge.score), reverse=True)
                edges = heapq.merge(edges, parent_parent_edges, key = lambda edge: abs_id(edge.score), reverse=True)

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        self.nodes['logits'].in_graph = any(parent_edge.in_graph for parent_edge in self.nodes['logits'].parent_edges)

        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue 
            
            if any(child_edge.in_graph for child_edge in node.child_edges):
                node.in_graph = True
            else:
                if prune_childless:
                    node.in_graph = False
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else: 
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True 
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if not isinstance(node, InputNode) and node.in_graph and not any(parent_edge.in_graph for parent_edge in node.parent_edges):
                    node.in_graph = False 
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False


    @classmethod
    def from_model(cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict], neuron_level: bool = False):
        graph = Graph()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        else:
            graph.cfg = model_or_config
            
        neurons = torch.zeros(graph.cfg['d_model']) if neuron_level else None
        input_node = InputNode(neurons)
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head, neurons) for head in range(graph.cfg['n_heads'])]
            mlp_node = MLPNode(layer, neurons)
            
            for attn_node in attn_nodes: 
                graph.nodes[attn_node.name] = attn_node 
            graph.nodes[mlp_node.name] = mlp_node     
                                    
            if graph.cfg['parallel_attn_mlp']:
                for node in residual_stream:
                    for attn_node in attn_nodes:          
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)
                
                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:     
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)
                        
        logit_node = LogitNode(graph.cfg['n_layers'])
        for node in residual_stream:
            graph.add_edge(node, logit_node)
            
        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1)
        graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1

        return graph

    def edge_matrices(self): 
        edge_scores = torch.zeros((self.n_forward, self.n_backward))
        edges_in_graph = torch.zeros((self.n_forward, self.n_backward)).bool()
        for edge in self.edges.values():
            edge_scores[self.forward_index(edge.parent, attn_slice=False), self.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = edge.score
            edges_in_graph[self.forward_index(edge.parent, attn_slice=False), self.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = edge.in_graph
            
        return edge_scores, edges_in_graph


    def to_json(self, filename: str, neurons: bool = False):
        # non serializable info
        d = {'cfg':self.cfg, 'nodes': {str(name): bool(node.in_graph) for name, node in self.nodes.items()}, 'edges':{str(name): {'score': None if edge.score is None else float(edge.score), 'in_graph': bool(edge.in_graph)} for name, edge in self.edges.items()}}
        d['node_scores'] = {str(name): node.score for name, node in self.nodes.items() if node.score is not None}
        if neurons:
            d['neurons'] = {str(name): node.neurons.tolist() for name, node in self.nodes.items() if node.neurons is not None}
            d['neuron_scores'] = {str(name): node.neuron_scores.tolist() for name, node in self.nodes.items() if node.neuron_scores is not None}
        with open(filename, 'w') as f:
            json.dump(d, f)
            
            
    def to_pt(self, filename: str, neurons: bool = False):
        src_nodes = {node.name: node.in_graph for node in self.nodes.values() if not isinstance(node, LogitNode)}
        dst_nodes = self.get_dst_nodes()
        edge_scores, edges_in_graph = self.edge_matrices()
        d = {'cfg':self.cfg, 'src_nodes': src_nodes, 'dst_nodes': dst_nodes, 'edges': edge_scores, 'edges_in_graph': edges_in_graph}
        
        if all(node.score is not None for node in self.nodes.values() if not isinstance(node, LogitNode)):
            d['node_scores'] = torch.tensor([node.score for node in self.nodes.values()  if not isinstance(node, LogitNode)])
            
        if neurons:
            d['neurons'] = torch.stack([node.neurons if node.neurons is not None else torch.ones(self.cfg['d_model']) for node in self.nodes.values() if not isinstance(node, LogitNode)])
            d['neuron_scores'] = torch.stack([node.neuron_scores if node.neuron_scores is not None else torch.zeros(self.cfg['d_model']) for node in self.nodes.values() if not isinstance(node, LogitNode)])
        torch.save(d, filename)

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str="dot",
        seed: Optional[int] = None
    ) -> pgv.AGraph:
        """
        Colorscheme: a cmap colorscheme
        """
        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(node.name, 
                        fillcolor=colors[node.name], 
                        color="black", 
                        style="filled, rounded",
                        shape="box", 
                        fontname="Helvetica",
                        )

        scores = self.scores().abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if edge.score is None else edge.score
                normalized_score = (abs(score) - min_score) / (max_score - min_score) if max_score != min_score else abs(score)
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(edge.parent.name,
                        edge.child.name,
                        penwidth=str(penwidth),
                        color=edge.get_color(),
                        )
        return g

    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (set(self.edges.keys()) == set(other.edges.keys()))
        if not keys_equal:
            return False
        
        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False 
            
        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(edge.score, other.edges[name].score):
                return False
        return True