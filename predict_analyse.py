import os
import json
import torch
from torchvision import transforms
from PIL import Image
import io

from model import swin_tiny_patch4_window7_224 as create_model


def predict_binary_images(binary_images, weights_path, num_classes=10, device='cuda:0'):
    # 设置设备为GPU或CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    img_size = 224
    # 定义数据预处理方式
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建模型并加载预训练权重
    model = create_model(num_classes=num_classes).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file: '{weights_path}' not found.")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 读取类别标签文件
    json_path = './class_indices.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File: '{json_path}' not found.")

    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    model.eval()

    predictions = []
    class_count = {}

    # 验证过程
    with torch.no_grad():
        for binary_image in binary_images:
            # 将二进制数据转换为PIL图像
            image = Image.open(io.BytesIO(binary_image)).convert('RGB')
            image = data_transform(image).unsqueeze(0).to(device)

            # 进行预测
            pred = model(image)
            pred_class = torch.max(pred, dim=1)[1].item()

            # 获取类别名称
            class_name = class_indict[str(pred_class)]
            predictions.append(class_name)

            # 更新类别计数
            if class_name in class_count:
                class_count[class_name] += 1
            else:
                class_count[class_name] = 1

    return predictions, class_count


# 调用
if __name__ == '__main__':
    # 读取二进制图像数据的示例
    with open('image1.jpg', 'rb') as f:
        binary_image1 = f.read()
    with open('image2.jpg', 'rb') as f:
        binary_image2 = f.read()

    binary_images = [binary_image1, binary_image2]
    weights_path = './weights/model-172.pth'

    predicted_classes, class_count = predict_binary_images(binary_images, weights_path)
    for idx, class_name in enumerate(predicted_classes):
        print(f"Image {idx + 1}: Predicted class - {class_name}")

    print("\nClass count:")
    for class_name, count in class_count.items():
        print(f"{class_name}: {count}")
