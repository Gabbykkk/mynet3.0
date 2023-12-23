import math
import os

import numpy as np
import matplotlib.pyplot as plt


def visualize_output_target(output, target, channel_idx=0):
    # 将output转换为numpy数组
    output_np = output.detach().cpu().numpy()  # 分离计算图并将output从GPU转移到CPU并转换为numpy数组

    # 获取指定通道的output
    channel_output = output_np[0, channel_idx]  # 获取第一个样本的指定通道output

    # 获取target
    target = target.detach().cpu().numpy()  # 分离计算图并将target从GPU转移到CPU并转换为numpy数组

    # 可视化output和target
    plt.subplot(1, 2, 1)  # 创建子图，1行2列，第1个子图
    plt.imshow(channel_output, cmap='gray')  # 使用灰度色彩映射绘制output图像
    plt.title('Output')  # 设置子图标题

    plt.subplot(1, 2, 2)  # 创建子图，1行2列，第2个子图
    plt.imshow(target[0], cmap='gray')  # 使用灰度色彩映射绘制target图像
    plt.title('Target')  # 设置子图标题

    plt.show()  # 显示图像对比


def visualize_collect_diffuse(collect_output, diffuse_output):
    B, _, C = collect_output.shape
    _, C, H, W = diffuse_output.shape

    # 选择某个样本进行可视化，这里选择第一个样本
    collect_sample = collect_output[0]  # (H*W, C)
    diffuse_sample = diffuse_output[0]  # (C, H, W)

    # 可视化collect_output和diffuse_output
    fig, axes = plt.subplots(2, C, figsize=(12, 6))

    # 可视化collect_output
    for c in range(C):
        collect_map = collect_sample[:, c].cpu().numpy().reshape(H, W)
        axes[0, c].imshow(collect_map, cmap='gray')
        axes[0, c].set_title(f'Collect {c + 1}')

    # 可视化diffuse_output
    for c in range(C):
        diffuse_map = diffuse_sample[c].cpu().numpy()
        axes[1, c].imshow(diffuse_map, cmap='gray')
        axes[1, c].set_title(f'Diffuse {c + 1}')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
    plt.show()


def plot_attention_heatmaps(attn_col_logit, attn_dif_logit):
    num_heads = 1
    batch_size, _, hw = attn_col_logit.shape
    height = int(math.sqrt(hw))
    width = int(math.sqrt(hw))

    # 创建保存图形的文件夹
    save_dir = f"./collect_diffuse/"  # 替换为实际的文件夹路径
    os.makedirs(save_dir, exist_ok=True)

    for i in range(20):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = axes.flatten()  # 展平为一维数组
        attn_col_logit_single = attn_col_logit[i].reshape(height, width)
        attn_dif_logit_single = attn_dif_logit[i].reshape(height, width)

        ax1 = axes[0]
        ax2 = axes[1]
        ax1.set_title(f"Sampler {i + 1}, Collect")
        ax2.set_title(f"Sampler {i + 1}, Diffuse")
        im1 = ax1.imshow(attn_col_logit_single.cpu().detach().numpy(), cmap='hot', aspect='auto',
                         extent=[0, width, 0, height])
        im2 = ax2.imshow(attn_dif_logit_single.cpu().detach().numpy(), cmap='hot', aspect='auto',
                         extent=[0, width, 0, height])

        fig.colorbar(im1, ax=ax1, orientation='vertical')
        fig.colorbar(im2, ax=ax2, orientation='vertical')

        plt.suptitle(f"Figure{i + 1}")  # 设置图形的名称

        plt.tight_layout()

        # 保存图形到文件夹
        file_name = f"figure_{i + 1}.png"  # 图片的文件名
        save_path = os.path.join(save_dir, file_name)  # 图片的完整路径
        plt.savefig(save_path)
        # plt.show()

        plt.close(fig)  # 关闭当前图形，释放内存
