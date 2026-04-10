import os
from pathlib import Path
import tiktoken
from tqdm import tqdm

MD_DIR = Path("data/thutuchanhchinh/markdown_json")

def analyze():
    print("⏳ Đang thống kê lượng Token thực tế trên các file Markdown...")
    if not MD_DIR.exists():
        print(f"Lỗi: Thư mục {MD_DIR} không tồn tại!")
        return
        
    md_files = list(MD_DIR.rglob("*.md"))
    
    # Dùng tokenizer phổ biến nhất của OpenAI (gpt-4 / text-embedding-ada-002)
    enc = tiktoken.get_encoding("cl100k_base")
    
    token_counts = []
    
    for file_path in tqdm(md_files, desc="Đang phân tích Token/File"):
        text = file_path.read_text(encoding="utf-8")
        tokens = enc.encode(text)
        token_counts.append((len(tokens), file_path.parent.name + "/" + file_path.name))
        
    if not token_counts:
        print("Không có tệp tin Markdown nào.")
        return
        
    # Sắp xếp để lấy min, max
    token_counts.sort(key=lambda x: x[0])
    
    min_tokens = token_counts[0]
    max_tokens = token_counts[-1]
    avg_tokens = sum(t[0] for t in token_counts) / len(token_counts)
    
    print("\n" + "=" * 55)
    print("📊 BÁO CÁO THỐNG KÊ LƯỢNG TOKEN/TÀI LIÊU (cl100k_base)")
    print("=" * 55)
    print(f"🎯 Tổng số file đem phân tích:  {len(token_counts)}")
    print(f"📐 Trung bình (Average):        {avg_tokens:.2f} tokens/file")
    print(f"🔴 Nhỏ nhất (Min):              {min_tokens[0]} tokens (File: {min_tokens[1]})")
    print(f"🔴 Lớn nhất (Max):              {max_tokens[0]} tokens (File: {max_tokens[1]})")
    
    # Xem phân phối nhanh (Distribution)
    bin_500 = len([t for t in token_counts if t[0] <= 500])
    bin_1000 = len([t for t in token_counts if 500 < t[0] <= 1000])
    bin_2000 = len([t for t in token_counts if 1000 < t[0] <= 2000])
    bin_4000 = len([t for t in token_counts if 2000 < t[0] <= 4000])
    bin_higher = len([t for t in token_counts if t[0] > 4000])
    
    print("-" * 55)
    print("📉 PHÂN PHỐI LƯỢNG TOKEN TRÊN TOÀN BỘ DATASET:")
    print(f" - [ Siêu Ngắn ] Dưới 500 tokens:      {bin_500} files")
    print(f" - [ Ngắn      ] Tới 1,000 tokens:     {bin_1000} files")
    print(f" - [ Vừa       ] Tới 2,000 tokens:     {bin_2000} files")
    print(f" - [ Dài       ] Tới 4,000 tokens:     {bin_4000} files")
    print(f" - [ Siêu Dài  ] Trên 4,000 tokens:    {bin_higher} files")
    print("=" * 55)

if __name__ == "__main__":
    analyze()
