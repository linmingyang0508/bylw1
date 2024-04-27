# 加载和准备数据集
# 音频处理和特征提取
# 重采样和格式转换

import json
import os
from datasets import load_dataset
import torchaudio
from transformers import Wav2Vec2Processor

# 预处理音频文件并创建数据集
def create_dataset(audio_dir, label_dir, output_json_path):
    data_records = []
    for audio_filename in os.listdir(audio_dir):
        if audio_filename.endswith(".wav"):
            audio_path = os.path.join(audio_dir, audio_filename)
            base_filename = os.path.splitext(audio_filename)[0]
            label_path = os.path.join(label_dir, base_filename + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as label_file:
                    label_text = label_file.read().strip()
                data_records.append({"path": audio_path, "text": label_text})
    with open(output_json_path, "w") as json_file:
        json.dump(data_records, json_file, indent=4, ensure_ascii=False)

# 指定文件夹路径
audio_dir = "C:\\Users\\ACER\\Desktop\\vhf_transformer\\data\\audio_dir"
label_dir = "C:\\Users\\ACER\\Desktop\\vhf_transformer\\data\\labels"
output_json_path = "C:\\Users\\ACER\\Desktop\\vhf_transformer\\data\\processed_data\\dataset.json"

# 创建数据集
create_dataset(audio_dir, label_dir, output_json_path)

# 加载数据集
dataset = load_dataset('json', data_files=output_json_path)

# 音频处理和特征提取

import torchaudio
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch

# 假设 `dataset` 是你加载并预处理过的数据集
dataset = dataset.map(speech_file_to_array_fn, remove_columns=["path", "text"])

# 应用重采样和格式转换
# dataset = dataset.map(resample_and_convert_to_features, remove_columns=["speech", "sampling_rate", "target_text"])
import numpy as np
from torchaudio.transforms import Resample

# 假设我们的模型期望的采样率是16kHz
target_sampling_rate = 16000

def resample_audio(batch):
    # 假设 `batch["sampling_rate"]` 是原始音频的采样率
    original_sampling_rate = batch["sampling_rate"]
    
    if original_sampling_rate != target_sampling_rate:
        # 初始化重采样器
        resampler = Resample(original_sampling_rate, target_sampling_rate)
        batch["speech"] = resampler(batch["speech"]).numpy()
    
    # 确保音频信号的范围在-1.0到1.0之间
    batch["speech"] = np.clip(batch["speech"], -1.0, 1.0)
    
    # 更新采样率为目标采样率
    batch["sampling_rate"] = target_sampling_rate
    
    return batch

# 应用音频重采样和格式转换
dataset = dataset.map(resample_audio)
