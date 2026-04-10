import os
from pathlib import Path

raw_dir = Path("data/thutuchanhchinh/raw")

def analyze():
    if not raw_dir.exists():
        print(f"Không tìm thấy thư mục: {raw_dir}")
        return
        
    agencies = [d for d in raw_dir.iterdir() if d.is_dir()]
    total_files = 0
    print(f"📊 THỐNG KÊ DỮ LIỆU TTHC (RAW) THEO BỘ/NGÀNH\n" + "-"*50)
    
    # Sort for better reading
    agencies = sorted(agencies, key=lambda x: x.name)
    
    for agency in agencies:
        files = list(agency.glob("*.txt"))
        count = len(files)
        total_files += count
        if count > 0:
            print(f" 📂 {agency.name}: {count} tài liệu")
            
    print("-" * 50)
    print(f"✅ TỔNG CỘNG: Tìm thấy {total_files} thủ tục nằm trong {len(agencies)} cơ quan.")

if __name__ == "__main__":
    analyze()
