import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#创建模型
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x,-300,300)))
def softmax(x):
    x_exp=np.exp(x-np.max(x,axis=1,keepdims=True))#softmax(x-c)=softmax(x),将样本做防溢出处理再指数化axis按行运算可以算多个样本
    x_sum=np.sum(x_exp,axis=1,keepdims=True)#求和
    return x_exp/x_sum
def sigmoid_derivative(x):
    #sigmoid求导
    return x*(1-x)
def one_hot_encode(y):
    return np.eye(10)[y.astype(int)]
class HandwritingModel:
    #初始化各种参数
    def __init__(self,input_size=784,hidden_size=128,output_size=10,learning_rate=0.8):
        #初始化各层维度
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        #初始化各层参数
        #输入->隐藏
        self.w1=np.random.normal(0,0.01,(input_size,hidden_size))
        self.b1=np.zeros((1,hidden_size))
        #隐藏->输出
        self.w2 = np.random.normal(0, 0.01, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
    #前向传播函数
    def forward(self,x):
        self.z1=np.dot(x,self.w1)+self.b1#输入层的线性结果
        self.a1=sigmoid(self.z1)#输入层的输出，隐藏层的输入
        self.z2=np.dot(self.a1,self.w2)+self.b2#隐藏层的线性输出
        self.a2=softmax(self.z2)#最终输出，激活用softmax
        return self.a2
    #反向激活函数，参数为输入的向量(即前向传播后的结果a2),和标签向量
    def backward(self,x,y):
        m=x.shape[0]#获取样本个数
        delta2=(self.a2-y)/m#交叉熵损失导数
        dw2=np.dot(self.a1.T,delta2)#求偏导，得出w2的梯度
        db2=np.sum(delta2,axis=0,keepdims=True)#b2的梯度
        #计算前两层的梯度
        delta1 = np.dot(delta2, self.w2.T) * sigmoid_derivative(self.a1)#链式求导
        dw1 = np.dot(x.T, delta1)#w1的梯度
        db1 = np.sum(delta1, axis=0, keepdims=True)#b1的梯度
        #更新参数
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
    #训练函数
    def train(self, x_train, y_train, epochs=30, batch_size=64):
        #获取样本个数
        m = x_train.shape[0]
        for epoch in range(epochs):#轮次循环
            indices = np.random.permutation(m)#生成随机排列
            #打乱数据集避免过拟合
            x_shuffle = x_train[indices]
            y_shuffle = y_train[indices]
            #分批次训练
            for batch in range(0,m,batch_size):
                x_batch=x_shuffle[batch:batch+batch_size]
                y_batch=y_shuffle[batch:batch+batch_size]
                self.forward(x_batch)
                self.backward(x_batch, y_batch)
            #记录训练成果
            y_pred = self.forward(x_train)
            loss = -np.mean(np.sum(y_train * np.log(y_pred + 1e-8), axis=1))  # 交叉熵损失（+1e-8防log(0)）
            # 打印进度
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f} ")
    #模型训练完后的测试函数
    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred,axis=1)#返回预测概率最大的索引
    #准确率评估函数
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        y_true=np.argmax(y,axis=1)#返回索引
        return np.mean(y_pred == y_true)#取平均得准确率
def load_data():
    #利用sklearn加载mnist数据集
    mnist=fetch_openml('mnist_784',cache=True)
    x=mnist.data.values.astype(np.float32)
    x/=255.0#归一化，经测试准确率可以从92到97
    y = mnist.target.values
    y_one_hot = one_hot_encode(y)#独热编码,将数字转为(10,)的向量
    print(y_one_hot.shape)
    x_train, x_test, y_train, y_test = train_test_split(x,y_one_hot,test_size=1/7,random_state=42)
    return x_train,x_test,y_train,y_test
if __name__=='__main__':
    #加载数据
    x_train, x_test, y_train, y_test=load_data()
    model=HandwritingModel()
    # 训练模型
    model.train(x_train, y_train)
    # 测试集评估
    test_acc = model.evaluate(x_test, y_test)
    print(f"\n测试集准确率：{test_acc:.4f}")