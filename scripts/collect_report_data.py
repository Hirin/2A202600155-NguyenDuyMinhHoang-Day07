"""
Script thu thập số liệu THỰC TẾ từ LM Studio để điền vào REPORT.md:
- Cosine similarity predictions dùng vietlegal-harrier-0.6b
- Chunking comparator chạy trên file TTHC thực (chunk_size=2000)
"""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(override=False)

from src.chunking import compute_similarity, ChunkingStrategyComparator
from src.embeddings import LMStudioEmbedder, MockEmbedder

# Thử dùng LM Studio thật, fallback về mock nếu chưa bật
print("🔌 Đang kết nối LM Studio...")
try:
    embedder = LMStudioEmbedder()
    # Test kết nối
    test_vec = embedder("test")
    print(f"✅ LM Studio OK — dim={len(test_vec)}, model={embedder._backend_name}")
except Exception as e:
    print(f"⚠️ LM Studio lỗi ({e}). Dùng MockEmbedder.")
    embedder = MockEmbedder()

# =============== SECTION 5: Similarity Predictions (5 cặp câu TTHC) ===============
pairs = [
    ("Thủ tục đăng kiểm tàu biển cần nộp hồ sơ gì?",
     "Hồ sơ đăng kiểm tàu biển gồm những giấy tờ nào?"),

    ("Cơ quan nào có thẩm quyền cấp giấy chứng nhận đăng kiểm?",
     "Chi cục Đăng kiểm là nơi tiếp nhận hồ sơ đăng kiểm."),

    ("Thời hạn giải quyết thủ tục là bao nhiêu ngày?",
     "Phí lệ phí nộp bao nhiêu tiền?"),

    ("Doanh nghiệp nộp hồ sơ trực tiếp tại cơ quan.",
     "Công ty nộp đơn trực tiếp ở bộ phận tiếp nhận."),

    ("Quy trình xem xét luận văn tiến sĩ tại trường đại học.",
     "Căn cứ pháp lý điều chỉnh an toàn tàu biển quốc tế."),
]

print("\n" + "=" * 65)
print("SECTION 5: COSINE SIMILARITY PREDICTIONS (vietlegal-harrier-0.6b)")
print("=" * 65)
results = []
for i, (a, b) in enumerate(pairs, 1):
    vec_a = embedder(a)
    vec_b = embedder(b)
    score = compute_similarity(vec_a, vec_b)
    results.append(score)
    print(f"\nPair {i}:")
    print(f"  A: {a}")
    print(f"  B: {b}")
    print(f"  Score: {score:.4f}  {'[HIGH]' if score > 0.6 else '[MED]' if score > 0.3 else '[LOW]'}")

# =============== SECTION 3: Chunking Comparator với chunk_size=2000 ===============
from pathlib import Path

# Dùng file TTHC dài vừa phải (~8000 chars) để kết quả dễ đọc
sample_files = [
    "data/thutuchanhchinh/markdown_json/BoXayDung/1.013225.md",
    "data/thutuchanhchinh/markdown_json/BoCongAn/1.001178.md",
]

print("\n" + "=" * 65)
print("SECTION 3: CHUNKING COMPARATOR (chunk_size=2000 chars)")
print("=" * 65)

for fpath in sample_files:
    p = Path(fpath)
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    print(f"\nFile: {p.name} — {len(text)} ký tự")
    print("-" * 50)
    result = ChunkingStrategyComparator().compare(text, chunk_size=2000)
    for name, stats in result.items():
        print(f"  [{name}] count={stats['count']}, avg_length={stats['avg_length']:.0f} chars")
