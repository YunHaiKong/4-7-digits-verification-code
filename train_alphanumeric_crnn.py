import numpy as np
import glob
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, LSTM, Input, Dropout, BatchNormalization, \
    Activation, Bidirectional, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, Callback
import cv2
from PIL import Image
import string
import os
import datetime
import tensorflow as tf

'''
第一步：加载数据集
'''
'''
1.1 定义数据集属性
'''
# 生成验证码的字符列表，包含数字和字母
CHAR_SET = string.digits + string.ascii_uppercase + string.ascii_lowercase
# 图片中每个字符可能属于几类
CHAR_SET_LEN = len(CHAR_SET)
# 输入图片统一为高32，宽128
img_size = (32, 128)
# 最大验证码长度
MAX_CAPTCHA_LEN = 7

'''
1.2 读取所有图片文件的路径
'''
# 获取文件夹中的图片数据
image_paths = []
data_dir = './train_alphanumeric'  # 数据目录

# 检查数据目录是否存在，如果不存在则创建
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"请先运行 generate_alphanumeric_captcha.py 生成训练数据")
    exit()

# 获取所有样本图片的路径
for i in range(4, 8):
    length_dir = os.path.join(data_dir, str(i))
    if os.path.exists(length_dir):
        image_paths += glob.glob(length_dir + '/*.png')

# 如果没有找到图片，提示用户生成数据
if len(image_paths) == 0:
    print(f"未找到训练数据，请先运行 generate_alphanumeric_captcha.py 生成训练数据")
    exit()

print(f"找到 {len(image_paths)} 张训练图片")
np.random.shuffle(image_paths)  # 将图片打乱

'''
1.3 分割训练集和验证集
'''
# 使用90%的数据用于训练，10%用于验证
total_samples = len(image_paths)
train_size = int(total_samples * 0.9)

train_img_paths = image_paths[:train_size]
test_img_paths = image_paths[train_size:]

# 验证集的构造
valid_size = min(4000, len(test_img_paths))  # 最多使用1000张图片作为验证集
valid_img_paths = test_img_paths[:valid_size]  # 验证样本的路径
x_valid = []  # 验证集的输入图片列表
y_valid = []  # 验证集图片对应的验证码

for valid_path in valid_img_paths:
    try:
        valid_img = np.array(Image.open(valid_path))  # 读取一张图片
        valid_img = cv2.resize(valid_img.astype(np.uint8), (img_size[1], img_size[0]))  # 缩放到定义好的尺寸
        valid_img = cv2.cvtColor(valid_img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
        x_valid.append(valid_img)  # 添加图片到列表中

        # 通过图片路径提取验证码
        valid_code = valid_path.split('/')[-1].split('\\')[-1].split('.')[0]  # 图片名是验证码
        y_valid.append(valid_code)  # 添加到验证码列表中
    except Exception as e:
        print(f"处理验证图片 {valid_path} 时出错: {e}")

# 图像像素归一化
x_valid = np.array(x_valid).reshape(-1, img_size[0], img_size[1], 1) / 255.0

'''
1.4 定义网络输出和验证码字符之间的转换方式
'''

def label_to_num(label):
    ''' 将文本转换为数字列表，例如'9aZ6'->[9,36,61,6] '''
    label_num = []
    for ch in label:
        label_num.append(CHAR_SET.find(ch))
    return np.array(label_num)


def num_to_label(num):
    ''' 将网络输出的序号转换为验证码字符串,例如[9,36,61,6,-1,-1,-1]->'9aZ6' '''
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += CHAR_SET[ch]
    return ret


'''
1.5 定义样本生成器
'''

def data_generator(image_paths, batch_size):
    ''' 给定图片路径列表，每次返回batch_size数量的图片x、验证码序号y、图片被切割的份数ctc_input_len、验证码字符串长度label_len、期望模型达到的ctc损失函数值y_ctc '''
    while True:
        '''1.5.1 定义要传入给网络的数据，其中x、y、label_len要通过遍历文件名得到'''
        x = []  # x代表本次batch的输入图片
        y = np.ones([batch_size, MAX_CAPTCHA_LEN]) * -1  # y代表本次batch的验证码序号
        label_len = np.zeros([batch_size, 1])  # 每张图片验证码字符串的长度
        ctc_input_len = np.ones([batch_size, 1]) * (15-2)  # 用于计算ctc损失函数的将图片切割的份数
        y_ctc = np.zeros([batch_size])  # ctc损失函数的期望值
        
        '''1.5.2 构造x、y、label_len'''
        # 随机从image_paths中抽取batch_size个图片路径，代表当前batch的图片
        batch_img_paths = np.random.choice(image_paths, batch_size)
        i = 0  # 当前图片在batch中的序号
        for img_path in batch_img_paths:
            try:
                # 读取一张图片，添加到batch的输入列表中
                img = np.array(Image.open(img_path))  # PIL读取RGB格式
                img = cv2.resize(img.astype(np.uint8), (img_size[1], img_size[0]))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x.append(img)

                # 根据图片名获取验证码，添加到batch的输入列表中
                code = img_path.split('/')[-1].split('\\')[-1].split('.')[0]  # 图片名是验证码
                y[i, 0:len(code)] = label_to_num(code)
                label_len[i] = len(code)
                i += 1
            except Exception as e:
                print(f"处理训练图片 {img_path} 时出错: {e}")
                # 如果处理图片出错，随机选择另一张图片
                new_img_path = np.random.choice(image_paths, 1)[0]
                img = np.array(Image.open(new_img_path))
                img = cv2.resize(img.astype(np.uint8), (img_size[1], img_size[0]))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x.append(img)

                code = new_img_path.split('/')[-1].split('\\')[-1].split('.')[0]
                y[i, 0:len(code)] = label_to_num(code)
                label_len[i] = len(code)
                i += 1

        # 图像像素归一化
        x = np.array(x).reshape(-1, img_size[0], img_size[1], 1) / 255.0
        yield [x, y, ctc_input_len, label_len], y_ctc


'''
第二步：构造网络
'''
'''
2.1 首先构造用于预测验证码的网络，这个网络包含输入层、卷积、双向LSTM、输出层。
    输出层输出形状为（batch_size，15，CHAR_SET_LEN+1）
'''
# 2.1.1 输入层，只包含1个输入，即被识别的图片
input_layer = Input(shape=(img_size[0], img_size[1], 1))
# 2.1.2 卷积部分 - 优化版本

# 第一个卷积块 - 32个滤波器
hidden_layer0 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_layer)
hidden_layer1 = BatchNormalization()(hidden_layer0)
hidden_layer2 = Activation('relu')(hidden_layer1)
hidden_layer00 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer2)
hidden_layer11 = BatchNormalization()(hidden_layer00)
hidden_layer22 = Activation('relu')(hidden_layer11)
# 添加Dropout防止过拟合
hidden_layer22 = Dropout(0.2)(hidden_layer22)
hidden_layer3 = MaxPooling2D(pool_size=(2, 2))(hidden_layer22)

# 第二个卷积块 - 64个滤波器
hidden_layer4 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer3)
hidden_layer5 = BatchNormalization()(hidden_layer4)
hidden_layer6 = Activation('relu')(hidden_layer5)
hidden_layer44 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer6)
hidden_layer55 = BatchNormalization()(hidden_layer44)
hidden_layer66 = Activation('relu')(hidden_layer55)
# 添加Dropout防止过拟合
hidden_layer66 = Dropout(0.2)(hidden_layer66)
hidden_layer7 = MaxPooling2D(pool_size=(2, 2))(hidden_layer66)

# 第三个卷积块 - 128个滤波器
hidden_layer8 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer7)
hidden_layer9 = BatchNormalization()(hidden_layer8)
hidden_layer10 = Activation('relu')(hidden_layer9)
hidden_layer88 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer10)
hidden_layer99 = BatchNormalization()(hidden_layer88)
hidden_layer110 = Activation('relu')(hidden_layer99)
# 添加Dropout防止过拟合
hidden_layer110 = Dropout(0.25)(hidden_layer110)
hidden_layer11 = MaxPooling2D(pool_size=(2, 2))(hidden_layer110)

# 第四个卷积块 - 256个滤波器
hidden_layer12 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer11)
hidden_layer13 = BatchNormalization()(hidden_layer12)
hidden_layer14 = Activation('relu')(hidden_layer13)
hidden_layer120 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(hidden_layer14)
hidden_layer130 = BatchNormalization()(hidden_layer120)
hidden_layer140 = Activation('relu')(hidden_layer130)
# 添加Dropout防止过拟合
hidden_layer140 = Dropout(0.25)(hidden_layer140)
# 在最后一个池化层中保持宽度，只减小高度，有助于保留水平方向的特征
hidden_layer15 = MaxPooling2D(pool_size=(2, 1))(hidden_layer140)

# 第五个卷积块 - 512个滤波器
hidden_layer16 = Conv2D(512, (2, 2), kernel_initializer='he_normal')(hidden_layer15)
hidden_layer17 = BatchNormalization()(hidden_layer16)
hidden_layer18 = Activation('relu')(hidden_layer17)
# 添加Dropout防止过拟合
hidden_layer18 = Dropout(0.3)(hidden_layer18)

# 2.1.3 将CNN的特征图转换为LSTM接受的序列化输入
# reshape为高度为1，宽度为15，深度为512的特征图.由于高度为1，所以高度的维度被去掉，所以这里等价于time_step = 15
hidden_layer19 = Reshape(target_shape=(15, 512))(hidden_layer18)

# 添加空间注意力机制，使用自定义层而不是Lambda层
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(1,),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # 计算注意力权重
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        # 应用注意力权重得到上下文向量
        context = x * a
        context = K.sum(context, axis=1, keepdims=True)
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[2])
        
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

class FeatureTileLayer(Layer):
    def __init__(self, tile_size, **kwargs):
        self.tile_size = tile_size
        super(FeatureTileLayer, self).__init__(**kwargs)
        
    def call(self, x):
        return K.tile(x, [1, self.tile_size, 1])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.tile_size, input_shape[2])
        
    def get_config(self):
        config = super(FeatureTileLayer, self).get_config()
        config.update({
            'tile_size': self.tile_size
        })
        return config

class FeatureConcatLayer(Layer):
    def __init__(self, **kwargs):
        super(FeatureConcatLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        return K.concatenate(inputs, axis=-1)
    
    def compute_output_shape(self, input_shapes):
        output_shape = list(input_shapes[0])
        output_shape[-1] = sum([shape[-1] for shape in input_shapes])
        return tuple(output_shape)
        
    def get_config(self):
        config = super(FeatureConcatLayer, self).get_config()
        return config

# 注册自定义层，确保模型保存和加载时能够正确识别
tf.keras.utils.get_custom_objects().update({
    'AttentionLayer': AttentionLayer,
    'FeatureTileLayer': FeatureTileLayer,
    'FeatureConcatLayer': FeatureConcatLayer
})

# 应用自定义注意力层
attention_layer = AttentionLayer()(hidden_layer19)
# 将注意力特征扩展到与原始特征相同的时间步长
attention_output = FeatureTileLayer(15)(attention_layer)
# 将注意力输出与原始特征连接
hidden_layer19_with_attention = FeatureConcatLayer()([hidden_layer19, attention_output])

# 纵向的维数从512+512缩小为256，减小计算量但保留更多信息
hidden_layer20 = Dense(256, activation='relu')(hidden_layer19_with_attention)
hidden_layer20 = Dropout(0.3)(hidden_layer20)

# 2.1.4 优化的双向LSTM部分
# 使用更高效的LSTM配置，添加recurrent_dropout防止过拟合
hidden_layer21 = Bidirectional(LSTM(192, return_sequences=True, 
                                  recurrent_dropout=0.2,
                                  kernel_initializer='he_normal'))(
    hidden_layer20)  # 输出维度为（batch,15,384）
hidden_layer21 = Dropout(0.25)(hidden_layer21)

hidden_layer22 = Bidirectional(LSTM(192, return_sequences=True,
                                  recurrent_dropout=0.2,
                                  kernel_initializer='he_normal'))(
    hidden_layer21)  # 输出维度为（batch,15,384）
hidden_layer22 = Dropout(0.25)(hidden_layer22)

# 2.1.5 优化的输出层
# 添加一个额外的Dense层增强特征表达
hidden_layer22_dense = Dense(256, activation='relu')(hidden_layer22)
hidden_layer22_dense = Dropout(0.2)(hidden_layer22_dense)

# 最终输出层
hidden_layer23 = Dense(CHAR_SET_LEN + 1)(hidden_layer22_dense)  #（batch_size, 15, CHAR_SET_LEN+1）
y_pred = Activation('softmax')(hidden_layer23)  #（batch_size, 15, CHAR_SET_LEN+1）

# 定义验证码预测网络模型的输入输出
model = Model(inputs=input_layer, outputs=y_pred)
# 查看网络结构
model.summary()

'''
2.2 构造用于计算ctc 损失函数的的子网络，这个网络接收验证码每个位置的真实类别、网络预测出的验证码长度、实际的验证码长度，
    输出层输出的是ctc损失函数的值
'''
'''
2.2.1 定义CTC损失函数的计算方式
'''


# 计算CTC损失函数，和损失函数计算子网络搭配使用
def ctc_lambda_func(input_list):
    y_true, y_pred, ctc_input_len, y_true_length = input_list  # 将传入参数拆解开
    y_pred = y_pred[:, 2:, :]  # 这里的2代表只使用LSTM第3个时刻开始的输出，因为第1和第2时刻的LSTM输出价值不大
    return K.ctc_batch_cost(y_true, y_pred, ctc_input_len, y_true_length)  # 调用keras自带的ctc计算方式


'''
2.2.2 定义CTC损失函数的输入和输出
'''
# CTC损失函数的计算要用额外的网络，网络一共有3个输入，分别是验证码每个位置的字符的类别、网络预测出的验证码长度、实际的验证码长度
y_true = labels = Input(shape=[MAX_CAPTCHA_LEN], dtype='float32')  # 长度为MAX_CAPTCHA_LEN的一维数组，代表每个位置的类别号
ctc_input_len = Input(shape=[1], dtype='int64')  # 只有一个值的一维数组, 代表传给ctc损失网络的序列长度
y_true_length = Input(shape=[1], dtype='int64')  # 只有一个值的一维数组,代表实际的验证码长度
ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [y_true, y_pred, ctc_input_len, y_true_length])  # 输出ctc损失函数值

'''
2.3 结合验证码识别网络和CTC损失函数网络，定义完整网络结构
'''
# 现在我们将识别验证码的网络和计算损失函数的网络拼起来，所以4个输入，第一个是识别网络的输入，后面三个是计算损失函数网络的输入。网络最终输出是损失函数值
model_final = Model(inputs=[input_layer, y_true, ctc_input_len, y_true_length], outputs=ctc_loss)
# 将这个大网络结构打印出来
model_final.summary()

'''第三步：模型编译'''
# 损失函数的计算已经在上面那个网络中完成了，但为了满足compile函数的语法要求，所以{}内随便定义了一个不会真正使用的损失函数
# "lambda y_true, y_output: y_output" 代表传入y_true和y_pred（也就是真实值和预测值），返回y_pred作为损失函数值（再次强调，并不会真正进行这个操作）
# 虽然没有真正使用，但要注意此处的损失函数名称'ctc',必须和ctc_loss=Lambda(...)那一行的name='ctc'保持一致

# 使用学习率调度器，随着训练进行逐渐降低学习率
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 使用固定学习率而不是学习率调度器，避免类型错误
initial_learning_rate = 1e-3

# 编译模型
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=initial_learning_rate))

'''
第四步：网络训练
'''
# 因为整个网络的输出是CTC损失函数值，CTC解码的结果（也就是预测出的验证码）被封装在K.ctc_batch_cost中没有暴露出来，keras内置的accuracy计算方式不适用于这种情况
# 只能自己实现对验证集准确率的计算。
num_epochs = 25  # 训练的epochs数量

'''
4.1 创建回调函数：TensorBoard、学习率调度器、早停策略和自定义准确率回调
'''
# 创建日志目录
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 确保日志目录存在
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# 创建TensorBoard回调
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch', write_graph=True)

# 创建学习率调度器回调 - 当验证损失不再改善时降低学习率
reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 创建早停策略 - 当验证损失不再改善时停止训练
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='loss',
    patience=8,  # 如果连续8个epoch没有改善就停止训练
    restore_best_weights=True,  # 恢复最佳权重
    verbose=1
)

# 创建自定义回调来记录准确率
class AccuracyCallback(Callback):
    def __init__(self, validation_data, log_dir, prediction_model, validation_steps=None):
        super(AccuracyCallback, self).__init__()
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.log_dir = log_dir
        self.prediction_model = prediction_model
        self.best_captcha_acc = 0  # 记录最佳验证码准确率
        
    def on_epoch_end(self, epoch, logs=None):
        # 获取验证集预测结果，使用传入的prediction_model而不是self.model(model_final)
        y_pred = self.prediction_model.predict(x_valid)
        # CTC解码
        decoded = K.get_value(
            K.ctc_decode(y_pred[:, 2:, ], input_length=np.ones(y_pred.shape[0]) * (y_pred.shape[1]-2),
                        greedy=True)[0][0])
        
        # 计算字符级和验证码级准确率
        total_char = 0
        correct_char = 0
        correct = 0
        
        # 用于错误样本分析
        error_samples = []
        
        for i in range(len(y_valid)):
            pr = num_to_label(decoded[i])
            tr = y_valid[i]
            total_char += len(tr)
            
            # 计算正确字符数
            correct_in_this_sample = 0
            
            # 使用最长公共子序列(LCS)算法计算匹配的字符数
            # 创建一个矩阵来跟踪匹配的字符
            m, n = len(tr), len(pr)
            # 初始化LCS矩阵
            lcs = [[0] * (n + 1) for _ in range(m + 1)]
            
            # 填充LCS矩阵
            for ii in range(1, m + 1):
                for jj in range(1, n + 1):
                    if tr[ii-1] == pr[jj-1]:  # 字符匹配
                        lcs[ii][jj] = lcs[ii-1][jj-1] + 1
                    else:  # 字符不匹配
                        lcs[ii][jj] = max(lcs[ii-1][jj], lcs[ii][jj-1])
            
            # 最长公共子序列的长度就是匹配的字符数
            correct_in_this_sample = lcs[m][n]
            correct_char += correct_in_this_sample
            
            # 计算此样本的字符准确率 - 使用正确匹配的字符数除以真实值和预测值长度的最大值
            # 这样当预测长度大于真实长度时，准确率会相应降低
            char_accuracy = correct_in_this_sample / max(len(tr), len(pr), 1)
            
            # 如果预测结果与真实值不同，记录错误样本
            if pr != tr:
                error_samples.append((tr, pr, f"{char_accuracy:.2f}"))
            else:
                correct += 1
        
        # 计算准确率
        char_acc = correct_char / total_char
        captcha_acc = correct / len(y_valid)
        
        # 记录最佳准确率
        if captcha_acc > self.best_captcha_acc:
            self.best_captcha_acc = captcha_acc
            # 当达到新的最佳准确率时保存模型
            self.prediction_model.save('best_captcha_alphanumeric_crnn.h5')
            print(f'\n新的最佳模型已保存! 验证码准确率: {captcha_acc:.4f}')
        
        # 将准确率添加到日志中
        logs = logs or {}
        logs['val_char_accuracy'] = char_acc
        logs['val_captcha_accuracy'] = captcha_acc
        
        # 使用TensorBoard记录自定义指标
        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.scalar('val_char_accuracy', char_acc, step=epoch)
            tf.summary.scalar('val_captcha_accuracy', captcha_acc, step=epoch)
            tf.summary.scalar('best_captcha_accuracy', self.best_captcha_acc, step=epoch)
            
        print(f'\n验证集 - 字符准确率: {char_acc:.4f}, 验证码准确率: {captcha_acc:.4f}, 最佳准确率: {self.best_captcha_acc:.4f}')
        
        # 显示错误样本分析
        if len(error_samples) > 0:
            print("\n错误样本分析（前10个）:")
            print("真实值 -> 预测值 (字符准确率)")
            for i, (true_val, pred_val, acc) in enumerate(error_samples[:10]):
                print(f"{true_val} -> {pred_val} ({acc})")


# 创建准确率回调实例
accuracy_callback = AccuracyCallback(validation_data=x_valid, log_dir=log_dir, prediction_model=model)

'''
4.2 数据增强和训练
'''
# 定义数据增强函数
def apply_random_augmentation(img):
    """对图像应用随机数据增强"""
    # 保存原始形状
    original_shape = img.shape
    
    # 确保图像是3D的 (高度, 宽度, 通道)
    if len(original_shape) == 3:
        # 提取图像数据，保持通道维度
        img_data = img[:, :, 0]  # 获取第一个通道的数据
    else:
        # 如果已经是2D，直接使用
        img_data = img
    
    # 随机亮度变化
    if np.random.random() < 0.3:
        delta = np.random.uniform(-0.1, 0.1)
        img_data = np.clip(img_data + delta, 0, 1)
    
    # 随机对比度变化
    if np.random.random() < 0.3:
        factor = np.random.uniform(0.8, 1.2)
        img_data = np.clip(img_data * factor, 0, 1)
    
    # 随机添加噪声
    if np.random.random() < 0.3:
        noise = np.random.normal(0, 0.01, img_data.shape)
        img_data = np.clip(img_data + noise, 0, 1)
    
    # 随机模糊
    if np.random.random() < 0.2:
        img_data = cv2.GaussianBlur(img_data, (3, 3), 0.5)
    
    # 恢复原始形状
    if len(original_shape) == 3:
        # 重新添加通道维度
        return img_data.reshape(original_shape[0], original_shape[1], original_shape[2])
    else:
        return img_data

# 修改数据生成器以包含数据增强
def augmented_data_generator(image_paths, batch_size):
    """带数据增强的数据生成器"""
    gen = data_generator(image_paths, batch_size)
    while True:
        [x, y, ctc_input_len, label_len], y_ctc = next(gen)
        # 对批次中的每张图片应用随机数据增强
        for i in range(x.shape[0]):
            if np.random.random() < 0.5:  # 50%的概率应用增强
                x[i] = apply_random_augmentation(x[i])
        yield [x, y, ctc_input_len, label_len], y_ctc

# 添加错误处理和调试信息
try:
    print("开始训练优化后的模型...")
    print(f"训练集大小: {len(train_img_paths)}, 验证集大小: {len(valid_img_paths)}")
    print(f"使用的回调函数: TensorBoard、学习率调度器、早停策略和准确率回调")
    print(f"已添加数据增强和Dropout防止过拟合")
    
    # 用训练集训练，添加所有回调函数
    model_final.fit(
        augmented_data_generator(train_img_paths, 128),  # 使用增强的数据生成器和更大的批次大小
        steps_per_epoch=300,  # 调整步数
        epochs=40,  # 增加最大训练轮数，但有早停机制
        verbose=2, 
        callbacks=[tensorboard_callback, accuracy_callback, reduce_lr, early_stopping]
    )
except Exception as e:
    print(f"训练过程中出错: {e}")
    import traceback
    traceback.print_exc()

'''
4.3 计算验证集准确率
'''
# 尝试加载最佳模型进行评估
try:
    print("\n加载最佳模型进行评估...")
    best_model_path = 'best_captcha_alphanumeric_crnn.h5'
    if os.path.exists(best_model_path):
        from tensorflow.keras.models import load_model
        # 加载模型时提供自定义层对象
        custom_objects = {
            'AttentionLayer': AttentionLayer,
            'FeatureTileLayer': FeatureTileLayer,
            'FeatureConcatLayer': FeatureConcatLayer
        }
        best_model = load_model(best_model_path, custom_objects=custom_objects)
        print(f"成功加载最佳模型: {best_model_path}")
        eval_model = best_model
    else:
        print("未找到最佳模型文件，使用当前模型进行评估")
        eval_model = model
except Exception as e:
    print(f"加载最佳模型出错: {e}，使用当前模型进行评估")
    eval_model = model

# 使用模型进行预测
y_valid_pred = eval_model.predict(x_valid)  # 获取网络输出

# 使用beam search解码提高准确率
beam_width = 5  # 设置beam search宽度
decoded = K.get_value(
    K.ctc_decode(y_valid_pred[:, 2:, ], 
               input_length=np.ones(y_valid_pred.shape[0]) * (y_valid_pred.shape[1]-2),
               greedy=False,  # 使用beam search而不是贪婪解码
               beam_width=beam_width)[0][0])

total_char = 0  # 验证集中包含的单个字符数
correct_char = 0  # 预测正确的单个字符数
correct = 0  # 识别正确的验证码个数。只有一个验证码中所有的字符都被预测正确，才被认为这个验证码识别正确

# 创建混淆矩阵分析
confusion_examples = []

for i in range(valid_size):  # 对第i张验证码图片
    pr = num_to_label(decoded[i])  # 将类别序号转换为字符串
    tr = y_valid[i]  # 获取真实的验证码字符串
    total_char += len(tr)  # 统计目前为止的已处理的字符个数

    # 因为预测出来的字符串长度可能与真实的长度有差异，为避免报错，哪个字符串更短，就以它的长度作为基准
    char_correct = 0
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:  # 如果两个相同位置的预测字符和真实字符相等
            correct_char += 1  # 预测正确的字符数量加一
            char_correct += 1
    
    # 计算此样本的字符准确率
    sample_char_acc = char_correct / len(tr) if len(tr) > 0 else 0
    
    if pr == tr:  # 如果预测字符串和真实字符串相等
        correct += 1  # 预测正确字符串的数量加1
    else:
        # 收集错误样本进行分析
        if len(confusion_examples) < 10:  # 只收集前10个错误样本
            confusion_examples.append((tr, pr, sample_char_acc))

# 打印评估结果
print('\n===== 模型评估结果 =====')
print('单个字符预测正确的概率 : %.2f%%' % (correct_char * 100 / total_char))
print('整个验证码预测正确的概率 : %.2f%%' % (correct * 100 / valid_size))

# 分析错误样本
if confusion_examples:
    print('\n错误样本分析（前10个）:')
    print('真实值 -> 预测值 (字符准确率)')
    for true, pred, char_acc in confusion_examples:
        print(f'{true} -> {pred} ({char_acc:.2f})')

# 保存最终模型
model.save('captcha_alphanumeric_crnn_final.h5')
print("\n最终模型已保存为 captcha_alphanumeric_crnn_final.h5")
print("最佳模型已保存为 best_captcha_alphanumeric_crnn.h5 (如果在训练过程中生成)")

# 打印TensorBoard查看指令
print(f"\n训练过程中的准确率曲线已保存到TensorBoard日志中")
print(f"可以通过以下命令启动TensorBoard查看训练曲线：")
print(f"tensorboard --logdir={log_dir}")
print(f"启动后，在浏览器中访问 http://localhost:6006 查看训练过程中的准确率曲线")

# 打印优化总结
print("\n===== 模型优化总结 ====")
print("1. 添加了Dropout层防止过拟合")
print("2. 使用了He正态分布初始化卷积核权重")
print("3. 添加了空间注意力机制提高特征提取能力")
print("4. 优化了LSTM层配置，增加了单元数量并添加了recurrent_dropout")
print("5. 使用了学习率调度器和早停策略")
print("6. 添加了数据增强提高模型泛化能力")
print("7. 使用了beam search解码提高识别准确率")
print("8. 自动保存最佳模型")
print("============================")