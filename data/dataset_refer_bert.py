import json
import os
import sys

import sng_parser
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

from tqdm import tqdm

# from add.ScenceGraph import  create_scence_graph, encode_scene_graph
from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


# 该类用于加载和处理与referring任务相关的数据集。继承自data.Dataset
class ReferDataset(data.Dataset):

    # 加载指代任务的数据集，并准备好输入句子和注意力掩码的表示形式，这些输入可以用于训练模型或预测
    def __init__(self,
                 args,  # 配置信息
                 image_transforms=None,  # 图像变换列表
                 target_transforms=None,  # 目标变换列表
                 split='train',  # 数据集划分方式
                 eval_mode=False):   # 是否为eval模式

        self.classes = []  # 空列表，存储数据集的类别信息
        self.image_transforms = image_transforms  # 存储图像的变换操作
        self.target_transform = target_transforms  # 目标的变换操作
        self.split = split  # 存储数据集的划分方式
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)  # 一个refer对象，用来处理referring任务的数据集。根据传入的参数，加载指定的数据集并进行初始化。
        self.ref_ids = self.refer.getRefIds(split=split)
        # self.sg_dir = os.path.join('./refer/SG', args.dataset, args.splitBy, self.split)
        # if not os.path.exists(self.sg_dir) or not os.listdir(self.sg_dir):
        #     os.makedirs(self.sg_dir, exist_ok=True)
        #     create_scence_graph(self.refer, self.ref_ids, self.sg_dir)

        self.max_tokens = 20  # 句子的最大标记数量

        ref_ids = self.refer.getRefIds(split=self.split)  # 根据划分方式获取referring数据的引用（reference）ID
        img_ids = self.refer.getImgIds(ref_ids)  # 根据reference ID 获取相应的图像ID

        all_imgs = self.refer.Imgs  # 数据集中的所有图像对象
        self.imgs = list(all_imgs[i] for i in img_ids)  # 根据图像ID获取相应的图像对象列表
        self.ref_ids = ref_ids  # 存储引用ID列表

        self.input_ids = []  # 存储输入句子的列表
        self.attention_masks = []  # 存储注意力掩码的列表
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)  # 一个BerTokenizer对象，用于对句子进行标记化和编码
        # self.model = BertModel.from_pretrained(args.bert_tokenizer).cuda()

        self.eval_mode = eval_mode  # 布尔值，表示是否为评估模式（测试模式）
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:  # 对于每一个引用ID，进行如下操作
            ref = self.refer.Refs[r]  # 获取引用对象

            sentences_for_ref = []  # 初始化用于存储该引用的句子的空列表
            attentions_for_ref = []  # 初始化用于存储该引用的注意力掩码的空列表

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):  # 对于每一个句子和句子ID的组合，进行以下操作
                sentence_raw = el['raw']  # 获取原始句子文本
                attention_mask = [0] * self.max_tokens  # 初始化注意力掩码的列表
                padded_input_ids = [0] * self.max_tokens  # 初始化填充后输入ID的列表

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)  # 对句子进行标记化和编码，得到输入ID（input_ids）

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]  # 将输入ID截断为最大标记数量

                padded_input_ids[:len(input_ids)] = input_ids  # 将截断后的输入ID填充到对应长度
                attention_mask[:len(input_ids)] = [1]*len(input_ids)  # 将注意力掩码填充到对应长度

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))  # 将填充后的输入ID转换为Pytorch张量，并添加到sentences_for_ref列表中
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))  # 将填充后的注意力掩码转换为Pytorch张量，并添加到attentions_for_ref列表中

            self.input_ids.append(sentences_for_ref)  # 将sentences_for_ref添加到input_ids列表中
            self.attention_masks.append(attentions_for_ref)  # 将attentions_for_ref添加到attention_masks列表中

    # 返回数据集中的类别信息
    def get_classes(self):
        return self.classes

    # 返回数据集中样本的数量，即引用对象的数量
    def __len__(self):
        return len(self.ref_ids)

    # 根据给定的索引index,获取数据集中的一个样本
    # 当运行ds[index]时，调用getitem方法
    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]  # 获取指定索引index对应的引用对象的ID
        this_img_id = self.refer.getImgIds(this_ref_id)  # 根据引用对象ID获取关联的图像ID
        this_img = self.refer.Imgs[this_img_id[0]]  # 从数据库中获取该图像的信息

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")  # 使用图像文件名加载图像文件，并将其转换为RGB模式的PIL图像对象

        ref = self.refer.loadRefs(this_ref_id)  # 加载指定引用对象的信息

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])  # 获取引用对象的分割掩码
        annot = np.zeros(ref_mask.shape)  # 根据引用对象的分割掩码，创建形状相同的全0注释数组
        annot[ref_mask == 1] = 1  # 将 annot 中对应于 ref_mask 中值为 1 的位置的元素设置为 1，即将 ref_mask 中被标记为引用目标的区域在 annot 中标注为 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")  # 将 annot 数组转换为 PIL 图像对象

        if self.image_transforms is not None:  # 如果定义了图像变换，则应用图像变换到图像和注释上
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:  # 如果处于评估模式，则将输入的嵌入向量self.input_ids和注意力掩码self.attention_masks组合为张量形式并返回
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:  # 如果不处于评估模式，随机选择一个句子choice_sent的嵌入向量和注意力掩码并返回
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        # node_features, adjacency_matrix = encode_scene_graph(this_ref_id, self.sg_dir, self.model, self.tokenizer)
        # node_features : [20,1,768]
        # adjacency_matrix : [4,4]

        return img, target, tensor_embeddings, attention_mask  # , node_features, adjacency_matrix
