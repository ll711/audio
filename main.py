import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- 1. 参数配置 ---
num_people = 20  # 说话人数量 (类别数)
name_people = 20  # 每个说话人的样本数
TOTAL_SAMPLES = num_people * name_people  # 400
FEATURE_DIMENSION = 40  # 假设使用 40 维特征 (如 MFCCs 或 Fbank)
AVG_FRAMES_PER_AUDIO = 100  # 假设每个音频平均有 100 帧


# --- 2. 模拟特征提取和数据对齐 ---

def simulate_feature_extraction():
    """
    *** ！！请将此函数替换为您真实的特征提取逻辑！！ ***

    此函数模拟从 400 个音频文件中提取特征和对应标签的过程。
    真实的实现需要使用 librosa 等库读取 .wav 文件，提取帧级别特征，
    并将每一帧与其所属的说话人标签对齐。

    返回：
    - X_data: 所有音频帧的特征向量集合 (N_total_frames, FEATURE_DIMENSION)
    - y_labels: 每一帧对应的说话人ID (N_total_frames,)
    """

    print("--- 模拟特征提取中 ---")

    # 初始化空的列表来存储所有帧数据和对应的标签
    all_features = []
    all_labels = []

    for speaker_id in range(num_people):
        for sample_id in range(name_people):
            # 模拟从一个音频样本中提取帧特征
            # 帧数在 80 到 120 之间随机变化
            num_frames = np.random.randint(
                AVG_FRAMES_PER_AUDIO - 20,
                AVG_FRAMES_PER_AUDIO + 20
            )

            # 模拟特征：随机生成 (num_frames, FEATURE_DIMENSION) 的数据
            features = np.random.rand(num_frames, FEATURE_DIMENSION).astype(np.float32)

            # 模拟标签：为所有这些帧分配当前说话人ID (0 到 19)
            labels = np.full((num_frames,), speaker_id, dtype=np.int32)

            all_features.append(features)
            all_labels.append(labels)

    # 将列表拼接成大的 NumPy 数组
    X_data = np.concatenate(all_features, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)

    print(f"模拟完成。总特征帧数: {len(X_data)}")
    print(f"X_data 形状: {X_data.shape}")
    print(f"y_labels 形状: {y_labels.shape}")
    print("--------------------")

    return X_data, y_labels


# 执行模拟特征提取
X_data, y_labels = simulate_feature_extraction()

# --- 3. 数据预处理和 8:2 划分 ---

# Step 3.1: 将说话人ID (0-19) 转换为 One-Hot 编码
y_one_hot = to_categorical(y_labels, num_classes=num_people)

print(f"One-Hot 标签形状: {y_one_hot.shape}")

# Step 3.2: 划分训练集和验证集 (基于帧的 80% 训练, 20% 验证)
# 注意：这种划分是随机的，对于某些场景可能需要按文件/说话人划分，
# 但对于传统的 DNN-HMM 声学模型帧分类，随机划分通常可以接受。
X_train, X_val, y_train, y_val = train_test_split(
    X_data,
    y_one_hot,
    test_size=0.2,
    random_state=42,  # 保证每次划分结果一致
    shuffle=True
)

print(f"\n--- 数据集划分结果 ---")
print(f"训练集特征形状 (80%): {X_train.shape}")
print(f"验证集特征形状 (20%): {X_val.shape}")
print("--------------------")


# --- 4. DNN 模型搭建 ---

def build_speaker_dnn_model(input_dim, num_classes):
    """
    搭建一个具有 BatchNormalization 和 Dropout 的 DNN 模型框架
    """
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),

        # 隐藏层 1
        Dense(1024),
        BatchNormalization(),  # Batch Norm
        tf.keras.layers.ReLU(),
        Dropout(0.5),  # Dropout

        # 隐藏层 2
        Dense(1024),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Dropout(0.5),

        # 隐藏层 3
        Dense(512),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        Dropout(0.5),

        # 输出层：20个类别，使用 Softmax
        Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 实例化模型
model = build_speaker_dnn_model(FEATURE_DIMENSION, num_people)

print("\n--- Keras DNN 模型结构 ---")
model.summary()
print("--------------------------\n")

# --- 5. 模型训练 ---

# 训练参数
EPOCHS = 50
BATCH_SIZE = 256  # 较大的 Batch Size 通常有助于 DNN 帧分类任务

print("--- 开始训练模型 ---")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1  # 显示训练进度
)

print("\n--- 模型训练完成 ---")