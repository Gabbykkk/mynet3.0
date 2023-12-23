image_path = 'demo/demo.jpg'  # 输入图像的路径
sentence = 'the most handsome guy'  # 用于推断的句子
weights = './checkpoints/refcoco.pth'  # 预训练权重文件的路径
device = 'cuda:0'

# pre-process the input image
from PIL import Image
import torchvision.transforms as T
import numpy as np
img = Image.open(image_path).convert("RGB")  # 用PIL库打开图像，并将其转换为RGB模式
img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization  将图像转换成Numpy数组，用于可视化
original_w, original_h = img.size  # PIL .size returns width first and height second  获取原始图像的宽度和高度

# 定义了一个图像转换的操作序列，包括调整大小，转换为Tensor和归一化
image_transforms = T.Compose(
    [
     T.Resize(480),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# 将图像应用转换操作，并添加一个维度作为Batch维度
img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
img = img.to(device)  # for inference (input)  将图像数据移到指定的设备上进行推断

# pre-process the raw sentence 对输入句子进行预处理
from bert.tokenization_bert import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)  # 使用Bertokenizer对句子进行编码，将其转换为一系列的标记
sentence_tokenized = sentence_tokenized[:20]  # if the sentence is longer than 20, then this truncates it to 20 words   截取前20个标记，并使用0进行填充，以保证长度一致
# pad the tokenized sentence
padded_sent_toks = [0] * 20
padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
# create a sentence token mask: 1 for real words; 0 for padded tokens 创建一个注意力掩码，其中真实的单词对应的位置为1，填充的位置为0
attention_mask = [0] * 20
attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
# convert lists to tensors  将处理后的标记和注意力掩码转换为Pytorch张量，并将他们移到指定的设备上
padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
padded_sent_toks = padded_sent_toks.to(device)  # for inference (input)
attention_mask = attention_mask.to(device)  # for inference (input)

# initialize model and load weights  初始化模型并加载权重的步骤
from bert.modeling_bert import BertModel
from lib import segmentation

# construct a mini args class; like from a config file
# 定义一个名为args的类，其中包含了一些模型的参数设置
class args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0

# 创建了一个'lavt'模型的实例，并加载了预训练的权重
single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
single_model.to(device)
# 创建了一个Bert模型的实例，并加载了预训练的权重
model_class = BertModel
single_bert_model = model_class.from_pretrained('bert-base-uncased')
single_bert_model.pooler = None

# 加载了之前保存的模型权重到相应的模型实例
checkpoint = torch.load(weights, map_location='cpu')
single_bert_model.load_state_dict(checkpoint['bert_model'])
single_model.load_state_dict(checkpoint['model'])
# 将模型实例转移到指定的设备上
model = single_model.to(device)
bert_model = single_bert_model.to(device)


# inference  推断过程的步骤
import torch.nn.functional as F
last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]  # 使用bert模型对输入句子进行推断，得到最后的隐藏状态
embedding = last_hidden_states.permute(0, 2, 1)  # 对隐藏状态进行一些变换，以适应模型的输入要求
output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))  # 将图像和句子的嵌入向量传递给模型，得到预测的输出
output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)  对输出进行一些后处理操作，包括取最大值，插值等
output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
output = output.squeeze()  # (orig_h, orig_w)
output = output.cpu().data.numpy()  # (orig_h, orig_w)  最后将输出转换为Numpy数组


# show/save results  展示/保存结果
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8  将输出转换为无符号8位整数类型
# Overlay the mask on the image  用此函数将输出的遮罩叠加到输入图像上，生成可视化结果
visualization = overlay_davis(img_ndarray, output)  # red
visualization = Image.fromarray(visualization)
# show the visualization
# visualization.show()
# Save the visualization  将可视化结果保存为图像文件
visualization.save('./demo/demo_result.jpg')




