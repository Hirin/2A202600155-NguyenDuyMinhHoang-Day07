import os
import random
import statistics
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import time

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from google import genai
except ImportError:
    genai = None

# Load environment variables
load_dotenv()

MD_DIR = Path("data/thutuchanhchinh/markdown_json")
SAMPLE_SIZE_GEMINI = 100

def analyze():
    print("⏳ Khởi tạo các tokenizers...")
    
    tokenizers = {}
    
    # OpenAI
    if tiktoken:
        tokenizers["openai"] = tiktoken.get_encoding("cl100k_base")
        print("✅ Đã load OpenAI tokenizer (cl100k_base).")
    else:
        print("❌ Chưa cài đặt thư viện 'tiktoken' (OpenAI).")
        
    # BERT Multilingual
    if AutoTokenizer:
        # Tắt cảnh báo log để tránh bị spam màn hình
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizers["bert"] = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        print("✅ Đã load mBERT tokenizer (bert-base-multilingual-cased).")
        try:
            tokenizers["harrier"] = AutoTokenizer.from_pretrained("mainguyen9/vietlegal-harrier-0.6b")
            print("✅ Đã load VietLegal-Harrier tokenizer (mainguyen9/vietlegal-harrier-0.6b).")
        except Exception as e:
            print(f"⚠️ Lỗi khi load VietLegal-Harrier tokenizer: {e}")
    else:
        print("❌ Chưa cài đặt thư viện 'transformers' (BERT/Harrier).")
        
    # Gemini Setup
    gemini_client = None
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if genai and gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
        print("✅ Đã load Google Gemini Client.")
    else:
        print("⚠️ Không tìm thấy GEMINI_API_KEY trong .env hoặc lỗi thư viện. Chỉ ước tính (estimated).")

    print("\n⏳ Đang thu thập danh sách files Markdown...")
    if not MD_DIR.exists():
        print(f"Lỗi: Thư mục {MD_DIR} không tồn tại!")
        return
        
    md_files = list(MD_DIR.rglob("*.md"))
    if not md_files:
        print("Không có tệp tin Markdown nào.")
        return
        
    total_files = len(md_files)
    print(f"🎯 Tổng số file đem phân tích:  {total_files}")
    
    # Thống kê tổng
    stats = {
        "files_count": total_files,
        "words": 0,
        "openai": 0,
        "bert": 0,
        "harrier": 0,
    }

    # Lưu lại để tính ratio
    gemini_sample_files = random.sample(md_files, min(SAMPLE_SIZE_GEMINI, total_files))
    openai_sampled_tokens = 0
    gemini_sampled_tokens = 0
    
    for file_path in tqdm(md_files, desc="Đang phân tích Token offline (OpenAI/BERT/Words)"):
        text = file_path.read_text(encoding="utf-8")
        
        # Đếm chữ (Words)
        stats["words"] += len(text.split())
        
        # OpenAI cl100k_base
        if "openai" in tokenizers:
            tokens_openai = tokenizers["openai"].encode(text)
            num_openai = len(tokens_openai)
            stats["openai"] += num_openai
            
            # Cập nhật số token sample cho Gemini nếu file này nằm trong mẫu
            if file_path in gemini_sample_files:
                openai_sampled_tokens += num_openai
                
        # BERT Multilingual & Harrier
        if "bert" in tokenizers:
            tokens_bert = tokenizers["bert"].encode(text, add_special_tokens=False)
            stats["bert"] += len(tokens_bert)
        if "harrier" in tokenizers:
            tokens_harrier = tokenizers["harrier"].encode(text, add_special_tokens=False)
            stats["harrier"] += len(tokens_harrier)

    # Nếu có client cho Gemini
    gemini_ratio = 1.0 # Mặc định coi như 1:1 với OpenAI nếu không gọi được API
    if gemini_client and openai_sampled_tokens > 0:
        print(f"\n⏳ Lấy mẫu {len(gemini_sample_files)} file qua Gemini API để tính tỷ lệ quy đổi...")
        successful_samples = 0
        
        for sf in tqdm(gemini_sample_files, desc="Gọi Gemini count_tokens API"):
            try:
                text = sf.read_text(encoding="utf-8")
                # Gọi API đếm token cho embed
                response = gemini_client.models.count_tokens(
                    model="models/embedding-001",
                    contents=text,
                )
                gemini_sampled_tokens += response.total_tokens
                successful_samples += 1
                # Rate limit sleep nhẹ nhàng (tránh 429)
                time.sleep(0.5) 
            except Exception as e:
                pass
                
        if successful_samples > 0 and openai_sampled_tokens > 0:
            gemini_ratio = gemini_sampled_tokens / openai_sampled_tokens
            print(f"✔️ Thành công {successful_samples}/{len(gemini_sample_files)} file. Tỷ lệ quy đổi (Gemini/OpenAI) = {gemini_ratio:.3f}")
        else:
            print("⚠️ Việc call Gemini API bị lỗi, sử dụng tỷ lệ quy ước mặc định.")
    
    
    # Tính toán kết quả cuối
    avg_words = stats["words"] / total_files
    avg_openai = stats["openai"] / total_files if stats["openai"] > 0 else 0
    avg_bert = stats["bert"] / total_files if stats["bert"] > 0 else 0
    avg_harrier = stats["harrier"] / total_files if stats["harrier"] > 0 else 0
    
    # Ước lượng tổng Gemini = Tổng OpenAI * Ratio
    est_total_gemini = int(stats["openai"] * gemini_ratio)
    avg_gemini = est_total_gemini / total_files if est_total_gemini > 0 else 0
    
    print("\n" + "=" * 65)
    print("📊 BÁO CÁO THỐNG KÊ LƯỢNG TOKEN TRÊN TOÀN BỘ DATASET TTHC")
    print("=" * 65)
    print(f"{'Phương pháp':<25} | {'Tổng Tokens':<15} | {'Trung bình/File':<15}")
    print("-" * 65)
    print(f"{'Từ (Words / Count)':<25} | {stats['words']:<15,d} | {avg_words:<15,.1f}")
    if "openai" in tokenizers:
        print(f"{'OpenAI (cl100k_base)':<25} | {stats['openai']:<15,d} | {avg_openai:<15,.1f}")
    if "bert" in tokenizers:
        print(f"{'mBERT (HuggingFace)':<25} | {stats['bert']:<15,d} | {avg_bert:<15,.1f}")
    if "harrier" in tokenizers:
        print(f"{'VietLegal-Harrier 0.6b':<25} | {stats['harrier']:<15,d} | {avg_harrier:<15,.1f}")
    if "openai" in tokenizers:
        gemini_label = "Gemini embedding-001" if gemini_client else "Gemini embed (Mặc định 1:1)"
        print(f"{gemini_label:<25} | {est_total_gemini:<15,d} | {avg_gemini:<15,.1f}")
    
    print("-" * 65)
    # Estimate costs using openAI embed-3-small (Cost is $0.02 / 1M tokens)
    if stats["openai"] > 0:
        openai_cost = (stats["openai"] / 1_000_000) * 0.02
        print(f"\n💡 Ước tính chi phí OpenAI text-embedding-3-small ($0.02/1M): ~${openai_cost:.4f}")
    
    if est_total_gemini > 0:
        # Giá Gemini embedding là miễn phí (Free tier) hoặc cực kì rẻ ~$0.000000 / 1M ở usage mode nhỏ
        print(f"💡 Ước tính chi phí Google Gemini embedding-001: Hoàn toàn miễn phí (Free Tier) hoặc Rất Rẻ (Pay-as-you-go)")
        
    print("=" * 65)


if __name__ == "__main__":
    analyze()
