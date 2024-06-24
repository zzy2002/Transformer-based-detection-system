import os
import json
import io
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model



def predict(bytes_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    #img_path = "./data/cs/ISIC_0024370.jpg"
    #assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #img = Image.open(img_path)
    img = Image.open(io.BytesIO(bytes_data))
    # [N, C, H, W]
    #展示
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # create model
    model = create_model(num_classes=10).to(device)
    #选用哪个权重
    model_weight_path = "./weights/model-172.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    #调用验证函数
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        #全部概率
        predict = torch.softmax(output, dim=0)
        #结果
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    #plt.title(print_res)
    for i in range(len(predict)):
    #    print("{:10}   ".format(str(i)))
        print("{:.3}".format(predict[i].numpy()))
    #print(predict_cla)

    # 将 JSON 字符串转换为字典
    # 提取字典中的值
    labels = list(class_indict.values())

    # 返回一个预测的结果序列,预测的病，病的标签
    return predict,class_indict[str(predict_cla)],labels



if __name__ == '__main__':
    img_path = "./data/cs/ISIC_0024370.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    predict(img)
