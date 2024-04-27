# 需要运行在NVIDIA CUDA、华为NPU或特定支持IPEX的XPU设备上，如果不是需要注释掉training_args = TrainingArguments中的fp16=True,
'''
需要额外安装：
pip install 'transformers[torch]'
pip install jiwer
'''

import os
from datasets import load_dataset, load_metric
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, TrainingArguments, Trainer

# 模型配置与初始化
processor = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    tokenizer=Wav2Vec2CTCTokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"),
)
model = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    attention_dropout=0.0,
    hidden_dropout=0.0, 
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

# 加载数据集
data_files = {"train": "../data/processed_data/dataset.json"}
dataset_split_not_found_err_msg = "Error: The specified dataset split was not found in the data files."
dataset = load_dataset(
    "json",
    data_files=data_files,
    keep_in_memory=True
)

# 拆分数据集为训练集和测试集
train_testvalid = dataset["train"].train_test_split(test_size=0.2)
train_data = train_testvalid["train"]
test_data = train_testvalid["test"]

# 处理音频长度不一致的问题

def prepare_dataset(examples):
    text = examples["text"]
    
    audios = []
    invalid_paths = []
    for rel_audio_path in examples["path"]:
        # 获取绝对路径
        abs_audio_path = os.path.join(os.path.dirname(__file__), "..", rel_audio_path)
        
        # 检查文件路径是否有效
        if os.path.isfile(abs_audio_path) and torchaudio.get_info(abs_audio_path):
            audios.append(torchaudio.load(abs_audio_path)[0])
        else:
            invalid_paths.append(abs_audio_path)
    
    # 如果没有有效的音频文件,直接返回
    if not audios:
        print(f"No valid audio files found. Invalid paths: {invalid_paths}")
        return None
    
    # 将音频列表转换为张量
    audios = torch.stack(audios)
    
    # 补白或裁剪音频,使其长度为16秒
    target_length = 16000 * 16  # 16秒,16kHz采样率
    if audios.shape[1] < target_length:
        audios = torch.nn.functional.pad(audios, (0, 0, 0, target_length - audios.shape[1]), "constant")
    else:
        audios = audios[:, :target_length]
    
    # 提取输入特征
    input_values = processor(audios, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
    
    # 解码标签为ID序列
    labels = [processor.tokenizer(text, max_length=model.config.max_length, truncation=True, padding="right").input_ids for text in text]
    labels = torch.tensor(labels)
    
    # 创建 attention_mask
    attention_mask = torch.ones_like(input_values, dtype=torch.long)
    
    return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}

# 应用处理函数
train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names, num_proc=4)
test_data = test_data.map(prepare_dataset, remove_columns=test_data.column_names, num_proc=4)

# 创建数据加载器
train_batch_size = 8
test_batch_size = 8

def custom_collate_fn(batch):
    # Extract input_values, attention_masks, and labels from the batch
    input_values = [item['input_values'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input_values and attention_masks to have the same length
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Since labels are a list of tensors of different lengths, we pad them manually
    # Find the longest label tensor
    max_label_length = max(len(label) for label in labels)
    # Pad each label tensor to match the longest one
    labels_padded = torch.full((len(labels), max_label_length), fill_value=processor.tokenizer.pad_token_id, dtype=torch.long)
    for i, label in enumerate(labels):
        labels_padded[i, :len(label)] = label

    return {
        "input_values": input_values,
        "attention_mask": attention_masks,
        "labels": labels_padded
    }

# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=train_batch_size, collate_fn=train_data.collate_fn, shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     test_data, batch_size=test_batch_size, collate_fn=test_data.collate_fn
# )
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=train_batch_size, collate_fn=custom_collate_fn, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=test_batch_size, collate_fn=custom_collate_fn
)


# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=test_batch_size,
    evaluation_strategy="steps",
    num_train_epochs=30,
    #fp16=True, # 需要在NVIDIA CUDA、华为NPU或特定支持IPEX的XPU设备上运行代码。
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_accumulation_steps=8,
    warmup_steps=1000,
    weight_decay=0.005,
    load_best_model_at_end=True,
)

# 初始化优化器
trainer = Trainer(
    model=model,
    data_collator=lambda x: x,
    args=training_args,
    compute_metrics=lambda pred: compute_metrics(pred.predictions, pred.label_ids, processor.tokenizer),
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
)

# 计算指标
wer_metric = load_metric("wer")
def compute_metrics(pred_ids, label_ids, tokenizer):
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 训练模型
trainer.train()

# 保存训练好的模型和处理器配置
trainer.save_model("../models/trained_models/wav2vec2-asr-chinese")
processor.save_pretrained("../models/trained_models/wav2vec2-asr-chinese")