import os
import json
import re
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/thutuchanhchinh/raw")
MD_OUTPUT_DIR = Path("data/thutuchanhchinh/markdown_json")

def extract_metadata(text_lines):
    """Trích xuất Meta data thành JSON object chuẩn hoá cho Vector Store"""
    meta_json = {
        "category": "thu_tuc_hanh_chinh",
    }
    tthc_title = ""
    
    for line in text_lines[:30]:
        match = re.match(r"^([^:]+):\s*(.*)", line)
        if match:
            key = match.group(1).strip()
            val = match.group(2).strip()
            
            if key == "Mã thủ tục": meta_json["ma_thu_tuc"] = val
            elif key == "Tên thủ tục": tthc_title = val
            elif key == "Cấp thực hiện": meta_json["cap_thuc_hien"] = val
            elif key == "Lĩnh vực": meta_json["linh_vuc"] = val
            elif key == "Số quyết định": meta_json["quyet_dinh"] = val
            elif key == "Cơ quan thực hiện": meta_json["co_quan_thuc_hien"] = val
            elif key == "Đối tượng thực hiện": 
                # Tách list các đối tượng thành mảng cho dễ filter
                meta_json["doi_tuong_thuc_hien"] = [x.strip() for x in val.split(',')]
    
    return meta_json, tthc_title

def convert_to_md_json(file_path: Path) -> None:
    text = file_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    
    # Lấy thông tin folder cơ quan từ đường dẫn (ví dụ: BoCongAn)
    agency_folder = file_path.parent.name
    code = file_path.stem
    
    metadata_json, tthc_title = extract_metadata(lines)
    metadata_json["agency_folder"] = agency_folder # Thêm trường ID cơ quan để dễ query filter
    
    # Bỏ qua phần Header / Metadata rác ở đầu
    start_body_idx = 15
    for i, line in enumerate(lines[:35]):
        if line.strip() and not re.match(r"^([^:]+):\s*(.*)", line) and "Chi tiết thủ tục hành chính" not in line:
            start_body_idx = i
            break
            
    body_text = "\n".join(lines[start_body_idx:])
    
    # Chuẩn hóa các mục lục chính thành Heading Markdown để cắt Chunk
    headers_keywords = [
        "Trình tự thực hiện", "Thành phần hồ sơ", "Yêu cầu, điều kiện thực hiện",
        "Cách thức thực hiện", "Đối tượng thực hiện", "Cơ quan thực hiện",
        "Kết quả thực hiện", "Căn cứ pháp lý", "Thời hạn giải quyết"
    ]
    for kw in headers_keywords:
        body_text = re.sub(rf"^({kw}:?.*)$", r"## \1", body_text, flags=re.MULTILINE|re.IGNORECASE)
    
    md_content = f"""```json
{json.dumps(metadata_json, ensure_ascii=False, indent=2)}
```

# {tthc_title or f"Thủ tục hành chính mã {code}"}

{body_text}
"""
    
    # Ghi ra thư mục MD xuất ra cùng cấu trúc folder
    out_dir = MD_OUTPUT_DIR / agency_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"{code}.md"
    out_file.write_text(md_content, encoding="utf-8")

def preprocess_all():
    print("🚀 Bắt đầu quá trình Preprocessing (Từ TEXT thô -> Markdown + JSON chuẩn)...")
    if not RAW_DIR.exists():
        print(f"Lỗi: Thư mục {RAW_DIR} không tồn tại!")
        return
        
    txt_files = list(RAW_DIR.rglob("*.txt"))
    print(f"🔍 Đã tìm thấy tổng cộng {len(txt_files)} tài liệu thủ tục hành chính thô.")
    
    for file_path in tqdm(txt_files, desc="Đang chuyển đổi & chuẩn hoá (ETL)", unit="file"):
        convert_to_md_json(file_path)
        
    print(f"\n✅ Hoàn thành phân tích! Toàn bộ 100% dữ liệu đã được Preprocessing.")
    print(f"📁 Dữ liệu cuối cùng lưu tại: {MD_OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_all()
