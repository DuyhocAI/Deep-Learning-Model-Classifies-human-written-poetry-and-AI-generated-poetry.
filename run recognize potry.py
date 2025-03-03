import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Đường dẫn tới model đã lưu
MODEL_DIR = r"D:\Bao_Duy\NLP\model_final"

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải tokenizer và model từ thư mục đã lưu
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def classify_poem(text):
    # Tiền xử lý văn bản đầu vào
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Dự đoán và tính xác suất
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    return prediction, confidence

# Định nghĩa bản đồ nhãn
label_map = {0: "Thơ do người viết", 1: "Thơ do AI viết"}

def read_multiline_input():
    print("Nhập đoạn thơ của bạn (gõ 'End' ở dòng riêng cuối cùng để kết thúc):")
    lines = []
    while True:
        line = input()
        if line.strip() == "End":
            break
        lines.append(line)
    return "\n".join(lines)

while True:
    poem = read_multiline_input()
    if poem.strip().lower() == "quit":
        break
    pred, conf = classify_poem(poem)
    print(f"Kết quả: {label_map[pred]} (độ tin cậy: {conf*100:.2f}%)")
