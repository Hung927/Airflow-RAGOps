import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

def find_target_sublist(my_list, target_sublist):
    target_length = len(target_sublist)
    for i in range(len(my_list)):
        if my_list[i:i + target_length] == target_sublist:
            return i, i + target_length

def extract_qa_pairs(data):
    """
    從SQuAD格式的數據中提取問題和答案對
    
    參數:
    raw_data - SQuAD格式的數據字典
    
    返回:
    qa_dict - 包含問題和答案對的字典，格式為 {question: answer}
    """
    qa_dict = {}    
    
    # 遍歷所有數據
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]
                
                # 只處理有可能回答的問題 (非不可能的問題)
                if not qa["is_impossible"]:
                    # 獲取第一個答案作為標準答案
                    if qa["answers"]:
                        answer = qa["answers"][0]["text"]
                        qa_dict[question] = answer
    
    return qa_dict

# 處理數據
raw_data = json.load(open("data/squad.json", "r"))
qa_pairs = extract_qa_pairs(raw_data)

# 輸出結果
for question, answer in qa_pairs.items():
    print(f"問題: {question}")
    print(f"答案: {answer}")
    print("-" * 50)

# 如果需要進一步處理或保存結果
with open('data/qa_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
    
# 下載 SQuAD 資料集
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json