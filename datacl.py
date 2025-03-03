import re
import os

# Dữ liệu mẫu
author = "Thế Lữ"
isAI = True  # True nếu là thơ do AI tạo ra
poemContent = """Ánh xuân lướt cỏ xuân tươi,
Bên rừng thổi sáo một hai Kim Đồng (pattern).
Tiếng đưa hiu hắt bên lòng,
Buồn ơi! Xa vắng, mênh mông (pattern) là buồn (pattern)...
Tiên Nga tóc xoã bên nguồn (pattern).
Hàng tùng rủ rỉ trên cồn đìu hiu;
Mây hồng ngừng lại sau đèo,
Mình cây nắng nhuộm, bóng chiều không đi.
Trời cao, xanh ngắt. - Ô kìa
Hai con hạc trắng bay về Bồng Lai.
Theo chim, tiếng sáo lên khơi,
Lại theo giòng suối bên người Tiên Nga.
Khi cao, vút tận mây mờ (patter),
Khi gần, vắt vẻo bên bờ (pattern) cây xanh,
Êm như lọt tiếng tơ tình,
Đẹp như Ngọc Nữ uốn mình trong không.
Thiên Thai thoảng gió mơ mòng,
Ngọc Chân buồn tưởng tiếng lòng xa bay..."""

# Hàm callback để thay thế mỗi lần khớp với biểu thức chính quy
def replace_rhyme(match):
    replace_rhyme.counter += 1
    return f"[Vần lặp {replace_rhyme.counter}]"

replace_rhyme.counter = 0  # Khởi tạo bộ đếm

# Thay thế tất cả các vị trí có "vần lặp" theo pattern bằng hàm callback
processed_poem = re.sub(r"(pattern)", replace_rhyme, poemContent)
# Đánh dấu kết thúc đoạn sau mỗi dòng
processed_poem = processed_poem.replace("\n", "\n[Kết thúc đoạn]\n")

# Ghép nội dung cuối cùng với các nhãn cần thiết
output = (
    f"[Tác giả: {author}]\n"
    f"[Thể loại: {'AI' if isAI else 'Người'}]\n"
    f"{processed_poem}\n"
    "[Kết thúc bài thơ]"
)

# Đảm bảo thư mục tồn tại trước khi ghi file
directory = r".\newdata\human\Thế Lữ"
os.makedirs(directory, exist_ok=True)

# Ghi nội dung ra file với đường dẫn đã
#  định nghĩa
file_path = os.path.join(directory, "Tiếng Sáo Thiên Thai.txt")
with open(file_path, "w", encoding="utf-8") as f:
    f.write(output)
