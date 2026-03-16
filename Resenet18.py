import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from PIL import Image

#将各种方法打包，对数据集进行各种操作
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(                #CIFAR10专属参数
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])
# 2. 加载数据集，在网上听说比较常用的就是CIFAR10数据集，看教程导入的
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
# CIFAR10类别名称，手打
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
# 3. 数据加载器（批量加载数据）
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 定义基础残差块（BasicBlock）
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积,首次用的一般的7x7的卷积核，效果并不好，各种调试加搜教程，确定应该是因为数据集的图片太小不符合224x224卷积核太大故改成3x3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化
        # 第二个3x3卷积（步幅固定为1）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # shortcut的下采样模块
        self.relu = nn.ReLU(inplace=True)  # 原地ReLU，节省内存

    def forward(self, x):
        identity = x
        # 主路径：conv1 -> bn1 -> relu -> conv2 -> bn2
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut路径：如果需要下采样则执行，否则直接使用原始输入
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        # 残差连接：主路径输出 + shortcut输出
        out += identity
        out = self.relu(out)  # 最后ReLU激活
        return out
# 定义ResNet主类
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        参数说明：
        - block: 残差块类型（这里用BasicBlock）
        - layers: 每个残差层的Block数量（ResNet18为[2,2,2,2]）
        - num_classes: 分类类别数（默认ImageNet的1000类）
        """
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始卷积后的通道数

        # 初始卷积层：7x7卷积 + 批归一化 + ReLU + 最大池化
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 4个残差层（layer1-layer4）
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化，输出1x1
        self.fc = nn.Linear(512, num_classes)

        # 初始化权重（保证模型训练稳定性）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建残差层（包含多个Block）
        参数说明：
        - block: 残差块类型
        - out_channels: 每个Block的输出通道数
        - blocks: 该层的Block数量
        - stride: 该层第一个Block的步幅
        """
        downsample = None

        # 当步幅≠1 或 输入通道≠输出通道时，需要下采样调整shortcut维度
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        # 构建该层的Block列表
        layers = []
        # 第一个Block（可能需要下采样）
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 更新输入通道数为当前输出通道数
        self.in_channels = out_channels
        # 剩余Block（步幅固定为1，无需下采样）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 4个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 池化 + 全连接
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平（保留batch维度）
        x = self.fc(x)

        return x
# 构建ResNet18实例的函数
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选GPU/CPU
model = resnet18(num_classes=len(class_names)).to(device)  # 分类数=类别列表长度
print(f"模型加载到设备：{device}")
print(f"模型分类数：{len(class_names)}")
# 1. 初始化TensorBoard（日志保存路径）
writer = SummaryWriter(log_dir='./runs/resnet18_cifar10')  # 日志会存在runs文件夹下

# 2. 定义损失函数、优化器、学习率调度器
criterion = nn.CrossEntropyLoss()  # 分类任务默认损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 3. 训练+验证循环
def train_model(model, train_loader, val_loader, epochs=20):
    best_val_acc = 0.0  # 保存最好的验证集准确率
    save_path = "./best_resnet18_model.pth"  # 模型保存路径

    for epoch in range(epochs):
        # ---------------------- 训练阶段 ----------------------
        model.train()  # 切换训练模式
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 统计训练指标
            train_loss += loss.item() * inputs.size(0)  # 累计损失，criterion默认返回批次的平均损失，所以得*批次数
            _, predicted = torch.max(outputs.data, 1)  # 预测类别（取概率最大的）
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 打印批次进度
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 计算训练集平均损失和准确率
        train_avg_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # ---------------------- 验证阶段 ----------------------
        model.eval()  # 切换验证模式（禁用Dropout/BatchNorm训练模式）
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # 验证时禁用梯度计算，节省内存
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证集平均损失和准确率
        val_avg_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # ---------------------- TensorBoard记录 ----------------------
        writer.add_scalars('Loss', {'train': train_avg_loss, 'val': val_avg_loss}, epoch+1)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch+1)

        # ---------------------- 打印epoch结果 ----------------------
        epoch_time = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{epochs}] Finished")
        print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Time Cost: {epoch_time:.2f}s")

        # ---------------------- 保存最好模型 ----------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f"Best model saved! Val Acc: {best_val_acc:.4f}\n")

        # 更新学习率
        scheduler.step()

    # 训练结束，关闭TensorBoard
    writer.close()
    print(f"Training Finished! Best Val Acc: {best_val_acc:.4f}")
    return model

# 开始训练（epochs设小一点，CPU跑10-20轮即可看到效果）
model = train_model(model, train_loader, val_loader, epochs=10)
def predict_image(image_path, model_path, class_names, device):
    """
    单张图片预测函数
    参数：
        image_path: 图片路径（如"./test.jpg"）
        model_path: 训练好的模型路径（如"./best_resnet18_model.pth"）
        class_names: 类别名称列表
        device: 设备（cpu/cuda）
    返回：
        pred_class: 预测类别名称
        pred_conf: 预测置信度（0-1）
    """
    # 1. 图片预处理（必须和训练时的val_transform一致！）
    transform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # 2. 加载并预处理图片
    try:
        image = Image.open(image_path).convert('RGB')  # 转为RGB（避免灰度图报错）
    except Exception as e:
        print(f"图片加载失败：{e}")
        return None, None

    image_tensor = transform(image).unsqueeze(0)  # 增加batch维度（从[3,224,224]→[1,3,224,224]）
    image_tensor = image_tensor.to(device)

    # 3. 加载模型（只加载权重，不加载优化器）
    model = resnet18(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 验证模式

    # 4. 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # 转为概率（0-1）
        pred_conf, pred_idx = torch.max(probabilities, 1)  # 取最大概率和索引
        pred_class = class_names[pred_idx.item()]
        pred_conf = pred_conf.item()

    # 5. 输出结果
    print(f"图片路径：{image_path}")
    print(f"预测类别：{pred_class}")
    print(f"置信度：{pred_conf:.4f}")
    return pred_class, pred_conf

