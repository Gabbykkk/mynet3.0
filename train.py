import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from functools import reduce
import operator
from torch.utils.tensorboard import SummaryWriter

from add.GAT import GAT
from add.MultiGCN import MultimodalGCN
from add.HLF import hlf
from add.PharseAtt import PhraseAttention
from bert.modeling_bert import BertModel
from lib import segmentation
import transforms as T
import utils
import numpy as np
import gc


class LossWeightOptim:
    """
    num_losses:要优化的损失函数数量
    smoothing_factor:计算指数移动平均的平滑因子
    """
    def __init__(self, num_losses, smoothing_factor=0.6):
        self.weights = [1.0/num_losses]*num_losses  # 初始化了损失权重，使得所有损失权重初始值相等
        self.smoothing_factor = smoothing_factor
        self.losses_moving_avg = [0] * num_losses  # 初始化损失的移动平均为0

    def update_weights(self, losses):  # 更新损失权重
        assert len(losses) == len(self.weights)  # 确保传入的损失列表长度等于当前的权重列表长度

        self.losses_moving_avg = [self.smoothing_factor*l + (1-self.smoothing_factor)*l_old
                                  for l, l_old in zip(losses, self.losses_moving_avg)]  # 计算每个损失的指数移动平均
        total_loss = sum(self.losses_moving_avg)  # 计算所有损失的移动平均的总和
        self.weights = [total_loss / (l + 1e-10) for l in self.losses_moving_avg]  # 根据每个损失的移动平均计算对应的权重。权重是总损失/每个损失的移动平均。为避免除0错误，在每个损失的移动平均中添加了一个很小的常数

        total_weight = sum(self.weights)  # 计算所有权重的总和
        self.weights = [w / total_weight for w in self.weights]  # 将每个权重除以总权重，以保证所有权重的总和为1

        return self.weights  # 返回更新后的权重列表


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def tml_loss(anchor, positive, negative):
    TML = torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean')
    return TML(anchor, positive, negative)


# 使用ReferDataset类创建一个数据集对象ds,并根据传入的参数进行初始化
def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,  # 指定数据集的划分方式，比如训练集、验证集或测试集。
                      image_transforms=transform,  # 图像的变换操作
                      target_transforms=None  # 目标数据的变换操作
                      )
    # ds数据集中的每个样本通常是一个元组或列表，包含以下元素：
    # 输入图像（img）：要处理或分析的图像。
    # 目标或注释（target）：对应图像的ground-truth分割掩码。
    # 张量嵌入（tensor_embeddings）：从BERT中获取的指代表达式的输入句子表示。
    # 注意力掩码（attention_mask）：对应输入句子的注意力掩码，指示要关注的标记位置。
    num_classes = 2  # 数据集的类别数量为2

    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


# 图像的变换操作
def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),  # 将输入图像调整大小为大小为args.img_size乘以args.img_size的正方形
                  T.ToTensor(),  # 将图像的表示从PIL图像或NumPy数组转换为PyTorch张量
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对张量图像进行归一化操作，标准化数据有利于提高网络性能
                  ]

    return T.Compose(transforms)  # 返回transforms变换列表


def evaluate(model, data_loader, bert_model, epoch):
    model.eval()  # 设为评估模式，模型将关闭一些具有随机性质的操作，比如dropout
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0  # 记录测试过的样本数量
    acc_ious = 0  # 累积预测结果的IoU
    # ---
    test_loss = 0  # 累积损失函数的值
    # ---

    # evaluation variables
    cum_I, cum_U = 0, 0  # 初始化用于计算overallIoU的变量
    eval_seg_iou_list = [.5, .6, .7, .8, .9]  # 包含了要计算的分割IoU的阈值
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)  # 记录当前阈值下预测正确的样本数量
    seg_total = 0  # 记录总共测试的样本数量
    mean_IoU = []  # 存储每一个样本的IoU值

    with torch.no_grad():  # 进入无梯度计算的环境，所有计算将不会被跟踪，节省内存并加快计算速度
        for data in metric_logger.log_every(data_loader, 100, header):  # 每经过100次迭代打印一次日志
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True), \
                                                target.cuda(non_blocking=True), \
                                                sentences.cuda(non_blocking=True), \
                                                attentions.cuda(non_blocking=True)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                # ---加
                l_pooler = bert_model(sentences, attention_mask=attentions)[1]
                all_hidden_states = bert_model(sentences, attention_mask=attentions)[2][:12]
                hidden_dim = bert_model.config.hidden_size  # bert-base:768
                hlf_model = hlf(hidden_dim, all_hidden_states).cuda()
                # pharAtt_model = PhraseAttention(hidden_dim)
                # mGCN_model = MultimodalGCN(768, 128, 0.5).cuda()
                # mGAT_model = GAT(768, 768)
                # ---加
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output, x_garan = model(image, embedding, l_pooler, hlf_model, all_hidden_states, l_mask=attentions)
            else:
                output = model(image, sentences, l_pooler, l_mask=attentions)

            # ---
            # target_l = torch.unsqueeze(target, 1)
            # loss_tml = TML(x_garan, target_l * x_garan, (1 - target_l) * x_garan)
            loss = criterion(output, target)  # 计算损失函数
            final_loss = loss
            test_loss += final_loss.item()  # 累积损失函数的值
            # ---

            iou, I, U = IoU(output, target)   # iou值，交集大小，并集大小
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I  # 更新overallIoU的累积值
            cum_U += U  # 更新overallIoU的累积值
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)  # 更新分割IoU的统计信息
            seg_total += 1  # 更新样本数量计数
        iou = acc_ious / total_its  #均IoU值

        # ---
        writer.add_scalar('test_loss', test_loss / len(data_loader), epoch)
        writer.add_scalar('iou', iou, epoch)
        # ---

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


# 一次训练
def train_one_epoch(model, criterion, optimizer, loss_weight_optim, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()  # 将模型设为训练模式 ，会启动模型中训练相关的操作，比如Dropout和BatchNormalization
    metric_logger = utils.MetricLogger(delimiter="  ")  # 创建一个MetricLogger对象用于记录训练中的指标，如损失和学习率
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))  # 在MetricLogger中添加一个名为lr的指标，用于记录平滑后的学习率
    header = 'Epoch: [{}]'.format(epoch)  # 定义一个字符串header，用于在日志中显示当前epoch数
    train_loss = 0  # 初始化训练损失
    total_its = 0  # 初始化总迭代次数
    # ---
    its_one_epoch = 0  # 一个epoch内的迭代次数
    # ---

    for data in metric_logger.log_every(data_loader, print_freq, header):  # 迭代训练data_loader中的每一个数据批次，使用log_every方法进行日志记录显示，print_freq是打印日志的频率，header是日志标题
        total_its += 1
        # print(image.size())  [20,3,320,320][B,C,H,W]
        # print(target.size()) # [20,320,320][B,H,W]
        # print(sentences.size())#  [20,1,20][B,1,T]
        # print(attentions.size()) # [20,1,20][B,1,T]
        image, target, sentences, attentions = data
        # print(node_features.size())  # torch.Size([20, 20, 1, 768])
        # print(adjacency_matrix.size())  # torch.Size([20, 20, 20])
        image, target, sentences, attentions = image.cuda(non_blocking=True), \
                                            target.cuda(non_blocking=True), \
                                            sentences.cuda(non_blocking=True), \
                                            attentions.cuda(non_blocking=True)  # 将迭代得到的数据和标签都转移到GPU上，non_blocking=True表示数据转移不会阻塞主线程
        # sentences和attentions都是形状为(batch_size, 1, sequence_length)的张量，其中第二个维度是维度为1的维度。
        # 这种情况通常出现在使用某些数据加载器或数据处理步骤时，可能会在维度上添加额外的维度，这可能会影响后续的操作。
        # 在这种情况下，通过squeeze(1)操作将维度为1的维度去除，使张量的形状变为(batch_size, sequence_length)，以便与后续的操作兼容。
        sentences = sentences.squeeze(1)  # 压缩sentences张量的第一个维度，即去除维度为1的维度,-->[B,T]
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768) [20,20,768][B,Ct,T]  最后一层隐藏状态
            # ---加
            l_pooler = bert_model(sentences, attention_mask=attentions)[1]  # [20,768] [B,Ct]  bert的池化表示
            # print(l_pooler.shape)
            all_hidden_states = bert_model(sentences, attention_mask=attentions)[2][:12]
            hidden_dim = bert_model.config.hidden_size
            hlf_model = hlf(hidden_dim, all_hidden_states).cuda()
            # pharAtt_model = PhraseAttention(hidden_dim)
            # mGCN_model = MultimodalGCN(768,128,0.5)
            # mGAT_model = GAT(768, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1) [20,20,1] 将注意力掩码扩展为3维张量
            # image:[20,3,320,320]
            output, x_garan = model(image, embedding, l_pooler, hlf_model, all_hidden_states, l_mask=attentions)  # model模型前向传播，得到output:[20,3,320,320]
        else:
            output = model(image, sentences, l_mask=attentions)

        target_l = torch.unsqueeze(target, 1)
        loss_tml = tml_loss(x_garan, target_l * x_garan, (1 - target_l) * x_garan)
        loss = criterion(output, target)  # 计算出输出output和目标target之间的损失  target:[20,320,320]

        weights = loss_weight_optim.update_weights([loss.item(), loss_tml.item()])

        final_loss = weights[0]*loss + weights[1]*loss_tml
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+  将优化器之前累积的梯度设置为0
        final_loss.backward()  # 根据反向传播算法计算出损失函数对模型各个参数的梯度值
        optimizer.step()  # 优化器更新模型的参数：优化器根据参数的梯度和学习率来计算更新量，参数量根据Adam规则进行调整，更新后的参数将代表模型在当前批次中的优化后状态
        lr_scheduler.step()  # 更新学习率：根据预定义的策略动态调整学习率，使得更好的收敛或跳出局部最优解

        # torch.cuda.synchronize()
        # ---
        total_train_step = its_one_epoch + epoch * len(data_loader)  # 记录训练过程中的总步数
        # ---
        train_loss += final_loss.item()  # 累计训练损失train_loss
        iterations += 1  # 更新迭代次数
        metric_logger.update(loss=final_loss.item(), lr=optimizer.param_groups[0]["lr"])  # 更新损失和学习率
        # ----加
        its_one_epoch += 1  # 当前epoch中的迭代次数
        if its_one_epoch % 500 == 0:  # 每500次迭代之后，将训练损失的平均值写入tensorboard中
            writer.add_scalar('train_loss', train_loss / its_one_epoch, total_train_step + 1)  # tensorboard
        # ----

        # plot_attention_heatmaps(attn_col, attn_dif)

        del image, target, sentences, attentions, final_loss, output, data  # 释放不再需要的变量
        if bert_model is not None:
            del last_hidden_states, embedding

        # 清除缓存以释放GPU显存
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()



def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    # num_samples = len(dataset)  # 训练集中的样本数：42404

    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)
    # num_samples_test = len(dataset_test)  # 测试集中的样本数：3811

    # batch sampler用于创建训练和测试数据集的批量采样器
    # 在多任务或分布式训练中，批量采样器用于确定每个进程在每个训练或测试步骤中要处理的数据子集
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
    #                                                                 shuffle=True)
    train_sampler = None
    # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    test_sampler = None

    # data loader  生成数据加载器，加载训练数据集并生成批量的数据样本。数据加载器在每个迭代中使用采样器从数据集中获取样本，然后将样本组成一个批次提供给模型进行训练或推断。
    # 通过创建数据加载器，您可以方便地遍历训练数据集，并以指定的批量大小获取数据样本。
    # 在训练过程中，可以使用数据加载器将数据样本输入到模型中进行训练。
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler,  # 采样器用于确定如何从数据集中选择样本进行训练或推断，决定了每个迭代中选择哪些样本，并且可以控制样本的顺序、重复性和分布
        num_workers=args.workers,  # 指定了用于加载数据的线程数
        pin_memory=args.pin_mem,  # 指定是否将数据加载到固定内存以加速数据传输
        drop_last=True)  # 当数据样本数量无法整除批次大小时，是否丢弃最后一个不完整的批次

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization  初始化了图像分割模型和bert模型，并移动到GPU上计算
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)  # 通过 args.model 参数指定的模型名称来获取 segmentation 模块中对应的模型类。
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    # single_model = model.module
    single_model = model

    if args.model != 'lavt_one':  # 如果模型不是lavt_one
        model_class = BertModel  # 指定模型类别为bertmodel
        bert_model = model_class.from_pretrained(args.ck_bert,output_hidden_states = True)  # 根据命令行参数中指定的ck_bert权重文件，加载对应的bert模型
        # bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)  # 将bert模型中的batch normalization层转换成同步batch normalization层，以支持多GPU并行训练
        # bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        # single_bert_model = bert_model.module
        single_bert_model = bert_model  # 用于后续单独访问bert模型的属性和方法
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=None)
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize  要优化的参数
    # 优化器中分别对这两个参数列表设置不同的学习率和权重衰减率，以进行参数更新优化
    backbone_no_decay = list()  # 不需要进行权重衰减的参数列表
    backbone_decay = list()  # 需要进行权重衰减的参数列表
    for name, m in single_model.backbone.named_parameters():  # 遍历，对每个参数的名称和参数本身进行如下操作
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:  # 若参数名称中含这三个，则加入到不需要权重衰减的列表中
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    # 将不同部分的模型参数分组，并为每个参数组设置不同的学习率和权重衰减值，以便在优化过程中对不同参数进行不同处理
    if args.model != 'lavt_one':  # 根据模型类型设置要优化的参数列表
        params_to_optimize = [  # params_to_optimize 是一个包含多个字典元素的列表。
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            # 每个字典元素代表一个参数组，其中包含以下键：params该参数组中需要进行优化的参数,weight_decay该参数组中参数的权重衰减值
            {'params': backbone_decay},  # 默认权重衰减设置
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # 获取 single_model.classifier 中需要进行优化（requireds_grad=True）的参数
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
            # 将 single_bert_model.encoder 中前10层的需要进行优化的参数连接在一起，形成一个统一的参数列表
        ]
    else:  # 不用
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer优化器，AdamW优化器，更新模型参数
    optimizer = torch.optim.AdamW(params_to_optimize,  # 需要优化的参数列表
                                  lr=args.lr,  # 学习率
                                  weight_decay=args.weight_decay,  # 权重衰减
                                  amsgrad=args.amsgrad  # 是否使用AMSGrad
                                  )

    # 初始化损失权重优化器
    loss_weight_optim = LossWeightOptim(num_losses=2)

    # learning rate scheduler  LambdaLR学习率调度器
    # 根据给定函数调整优化器的学习率
    # lambda函数根据当前迭代次数x来计算学习率的衰减因子，衰减因子为(1 - x / (len(data_loader) * args.epochs)) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()  # 开始训练的时间
    iterations = 0  # 迭代次数
    best_oIoU = -0.1  # 最佳交并比

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch + 1), args.epochs):  # 循环遍历每一个epoch,从上次中断的位置或者0开始，直到指定的总epoch数
        # data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, loss_weight_optim, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, epoch)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)  # 检查当前overallIoU是否优于之前的最佳oIoU
        if save_checkpoint:  # 如果是，表示当前oIoU最佳，
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:  # lavt则single_bert_model！=none
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}  # 创建字典，其中包含要保存的模型参数，优化器参数，当前epoch、命令行参数和学习率调度器参数
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(
                                                                args.try_id)))  # 将字典保存到指定文件路径
            best_oIoU = overallIoU  # 更新best_oIoU为i当前oIoU，作为新的最佳oIoU


    # summarize
    total_time = time.time() - start_time  # 计算从训练开始到当前代码执行位置的总耗时
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 将总耗时转换为格式化的字符串，以小时、分钟、秒的形式表示
    print('Training time {}'.format(total_time_str))  # 输出总耗时，展示整个训练时长


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    # utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    writer = SummaryWriter(args.tensorboard_path)
    main(args)
    writer.close()
