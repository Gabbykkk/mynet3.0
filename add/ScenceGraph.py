import argparse
import os

import sng_parser
import torch
from fastNLP import Vocabulary
from matplotlib import pyplot as plt
from tqdm import tqdm

from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer
from refer.refer import REFER
import json
import networkx as nx
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser()


def create_scence_graph(refer, ref_ids,sg_dir):
    """
    这个函数从 RefCoco 数据集中读取数据，并使用 sng_parser 对每个句子进行解析，生成场景图。场景图被保存为一个字典，其中包含对象和关系。这个字典被保存为一个 json 文件。
    """
    print('There are %s referred objects.' % len(ref_ids))

    # 使用 tqdm 来显示处理进度
    for ref_id in tqdm(ref_ids, desc="Processing images"):
        ref = refer.loadRefs(ref_id)[0]
        # refer.showRef(ref, seg_box='seg')
        scene_graph = {}
        scene_graph['objects'] = []
        scene_graph['relationships'] = []
        for sid, sent in enumerate(ref['sentences']):
            graph = sng_parser.parse(sent['sent'])
            for eni in graph['entities']:
                if eni['head'] not in scene_graph['objects']:
                    scene_graph['objects'].append(eni['head'])
            for rei in graph['relations']:
                temp = []
                temp.append(rei['subject'])
                temp.append(rei['relation'])
                temp.append(rei['object'])
                if temp not in scene_graph['relationships']:
                    scene_graph['relationships'].append(temp)
        # 将每个graph字典保存为一个单独的json文件
        with open(os.path.join(sg_dir, "ScenceGraph_{}.json".format(ref_id)), "w", encoding='utf-8') as f:
            f.write(json.dumps(scene_graph, ensure_ascii=False))
    print("\n已将数据集中全部图片处理成场景图")


# def create_vocab(sg_folder):
#     object_name_to_idx = {}
#     pred_name_to_idx = {}
#
#     # 遍历 SG 文件夹中的所有 JSON 文件
#     for filename in os.listdir(sg_folder):
#         if filename.endswith('.json'):
#             filepath = os.path.join(sg_folder, filename)
#             with open(filepath, 'r') as f:
#                 scene_graph = json.load(f)
#
#             for obj in scene_graph['objects']:
#                 if obj not in object_name_to_idx:
#                     object_name_to_idx[obj] = len(object_name_to_idx)
#             for s, p, o in scene_graph['relationships']:
#                 if p not in pred_name_to_idx:
#                     pred_name_to_idx[p] = len(pred_name_to_idx)
#
#     vocab = {
#         'object_name_to_idx': object_name_to_idx,
#         'pred_name_to_idx': pred_name_to_idx,
#     }
#
#     return vocab


def encode_scene_graph(ref_id, sg_dir, model, tokenizer):
    with open(os.path.join(sg_dir, 'ScenceGraph_{}.json'.format(ref_id)), 'r') as f:  # 打开指定场景图的文件
        scene_graph = json.load(f)

    MAX_NODES = 20  # Maximum number of nodes
    adjacency_matrix = torch.zeros((MAX_NODES, MAX_NODES))  # 初始化一个全零的邻接矩阵，大小为20x20

    for i, (s, p, o) in enumerate(scene_graph['relationships']):  # 遍历场景图中的所有关系
        if s < MAX_NODES and o < MAX_NODES:  # 如果关系的两个节点的索引都小于最大节点数。
            adjacency_matrix[s, o] = 1
            adjacency_matrix[o, s] = 1  # if the graph is undirected

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取节点的特征
    node_features = torch.zeros((MAX_NODES, 1, 768), device=device)  # 初始化一个全零张量
    for i, node in enumerate(scene_graph['objects']):
        if i >= MAX_NODES:
            break
        inputs = tokenizer(node, return_tensors='pt').to(device)
        outputs, _ = model(**inputs)
        node_features[i] = outputs[:, 0, :]  # 直接在大张量中存储节点特征

    return node_features, adjacency_matrix




# if __name__ == '__main__':
#     create_scence_graph()
#     args = parser.parse_args()



