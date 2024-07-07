import os

def load_qa_pairs(file_path):
    qa_pairs = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "::" in line:
                question, answer = line.strip().split("::")
                qa_pairs[question] = answer
    return qa_pairs

def find_best_match(question, qa_pairs):
    # 簡單的字符串匹配方法
    for q in qa_pairs:
        if question in q:
            return qa_pairs[q]
    return "對不起，我不明白你的問題。"

def main():
    file_path = 'questions_answers.txt'
    
    # 確認文件存在
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
