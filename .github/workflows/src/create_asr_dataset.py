import os
import json

# 定义音频和标签文件夹路径
audio_dir = "C:\\Users\\ACER\\Desktop\\vhf_transformer\\data\\audio_dir"
label_dir = "C:\\Users\\ACER\\Desktop\\vhf_transformer\\data\\labels"

# 定义输出JSON文件的路径
output_json_path = "C:\\Users\\ACER\\Desktop\\vhf_transformer\\data\\processed_data\\dataset1.json"

# 初始化一个列表来保存所有数据记录
data_records = []

# 遍历音频文件夹中的所有文件
for audio_filename in os.listdir(audio_dir):
    if audio_filename.endswith(".wav"):  # 确保只处理.wav文件
        # 构建完整的音频文件路径
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # 尝试找到对应的标签文件
        base_filename = os.path.splitext(audio_filename)[0]
        label_path = os.path.join(label_dir, base_filename + ".txt")
        
        # 检查标签文件是否存在
        if os.path.exists(label_path):
            # 读取标签文件内容
            with open(label_path, 'r') as label_file:
                label_text = label_file.read().strip()  # 删除可能的前后空白字符
                
            # 添加记录到列表
            data_records.append({"path": audio_path, "text": label_text})

# 写入JSON文件
with open(output_json_path, "w") as json_file:
    json.dump(data_records, json_file, indent=4, ensure_ascii=False)

print(f"JSON file created at {output_json_path} with {len(data_records)} records.")
