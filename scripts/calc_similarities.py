import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Đảm bảo đọc được file .env ở gốc
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "text-embedding-3-small"

pairs = [
    ("Làm thế nào để nộp đơn khởi kiện trực tuyến?", "Quy trình nộp hồ sơ khởi kiện qua cổng dịch vụ công của Tòa án."),
    ("Phiếu nhận xét đảng viên nơi cư trú dùng để làm gì?", "Đánh giá kết quả thực hiện nhiệm vụ của đảng viên tại địa phương."),
    ("Thời hạn giải quyết hồ sơ là bao nhiêu ngày?", "Lệ phí nộp đơn khởi kiện là bao nhiêu?"),
    ("Đảng uỷ xã tiếp nhận thông tin giới thiệu đảng viên.", "Cơ quan cấp xã xử lý hồ sơ sinh hoạt nơi cư trú."),
    ("Hướng dẫn cách làm món phở bò Nam Định.", "Hồ sơ khởi kiện cần những giấy tờ gì?"),
]

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    print(f"Algorithm: OpenAI {model}")
    print("-" * 50)
    
    for i, (a, b) in enumerate(pairs, 1):
        # Lấy embedding
        res_a = client.embeddings.create(input=[a], model=model)
        res_b = client.embeddings.create(input=[b], model=model)
        
        vec_a = res_a.data[0].embedding
        vec_b = res_b.data[0].embedding
        
        # Vì vector OpenAI đã được chuẩn hóa (unit vector), dot product chính là cosine similarity
        score = dot_product(vec_a, vec_b)
        print(f"Pair {i}: {score:.4f}")

if __name__ == "__main__":
    main()
