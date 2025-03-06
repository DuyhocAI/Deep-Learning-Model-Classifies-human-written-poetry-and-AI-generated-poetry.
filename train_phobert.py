import os
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Đường dẫn gốc của data
DATA_DIR_HUMAN = r".\data_train_val\human"
DATA_DIR_GPT = r".\data_train_val\GPT_4o3"

BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256  # Đặt lại độ dài sequence phù hợp với PhoBERT
MODEL_NAME = "vinai/phobert-base"  # PhoBERT-base (~110 triệu tham số)
OUTPUT_DIR = "./model_final"

# Early stopping parameters
PATIENCE = 8  # Số epoch không cải thiện tối đa
best_val_loss = float('inf')
patience_counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="./runs_phobert/poem_classification")

def load_poems_from_dir(root_dir):
    """
    Load tất cả file .txt từ các folder con trong root_dir.
    """
    poems = []
    pattern = os.path.join(root_dir, "*", "*.txt")
    for filepath in glob.glob(pattern):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            # Loại bỏ meta tag và dấu phân cách
            text = re.sub(r'\[.*?\]', '', text)
            text = text.strip()
            if text:
                poems.append(text)
    return poems

# Load dữ liệu thơ
human_poems = load_poems_from_dir(DATA_DIR_HUMAN)
ai_poems = load_poems_from_dir(DATA_DIR_GPT)

# Gán nhãn: 0 cho thơ của người, 1 cho thơ của AI
texts = human_poems + ai_poems
labels = [0] * len(human_poems) + [1] * len(ai_poems)

# Định nghĩa Dataset cho thơ
class PoemDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Sử dụng tokenizer từ PhoBERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = PoemDataset(texts, labels, tokenizer, MAX_SEQ_LENGTH)

# Chia dữ liệu thành train/validation (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Tạo cấu hình model với dropout cao hơn (0.3) và số nhãn = 2
# Đảm bảo type_vocab_size=1 cho Roberta/PhoBERT
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3,
    type_vocab_size=1
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tổng số tham số: {total_params:,}")
print(f"Số tham số huấn luyện được: {trainable_params:,}")

# Cài đặt optimizer với weight decay (regularization)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for batch in train_pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        global_step += 1
        epoch_loss += loss.item()
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if global_step % 10 == 0:
            writer.add_scalar("Loss/train", loss.item(), global_step)
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss trung bình: {avg_loss:.4f}")
    writer.add_scalar("Loss/epoch_train", avg_loss, epoch)
    
    # Đánh giá trên tập validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            logits = outputs.logits
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.4f} - Accuracy: {accuracy:.4f}")
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/val", accuracy, epoch)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping tại epoch {epoch+1} với validation loss: {avg_val_loss:.4f}")
            break

writer.close()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model và tokenizer đã được lưu tại: {OUTPUT_DIR}")
