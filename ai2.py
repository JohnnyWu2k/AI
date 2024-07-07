import os
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

def load_qa_pairs(file_path):
    qa_pairs = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "::" in line:
                question, answer = line.strip().split("::")
                qa_pairs[question] = answer
    return qa_pairs

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def find_best_match(question, qa_pairs):
    question = preprocess(question)
    question_embedding = get_embeddings(question)
    
    best_match = None
    best_similarity = float('-inf')

    for q in qa_pairs:
        q_preprocessed = preprocess(q)
        q_embedding = get_embeddings(q_preprocessed)
        
        # 計算餘弦相似度
        similarity = torch.nn.functional.cosine_similarity(question_embedding, q_embedding).item()
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = q

    return qa_pairs[best_match] if best_match else "對不起，我不明白你的問題。"

def main():
    file_path = 'questions_answers.txt'
    
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在。請確保文件位於當前目錄。")
        return

    qa_pairs = load_qa_pairs(file_path)

    print("你好！我是AI助手，有什麼問題可以問我。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['退出', '再見', 'bye']:
            print("AI助手: 再見！")
            break
        response = find_best_match(user_input, qa_pairs)
        print(f"AI助手: {response}")

if __name__ == "__main__":
    main()
