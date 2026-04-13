"""Section heading normalization map for Vietnamese administrative procedures (TTHC).

Maps Vietnamese section headings (lowercase, stripped) to canonical keys
used throughout the pipeline for filtering, routing, and citation.
"""
from __future__ import annotations


# Heading text (lowercase) → canonical section key
SECTION_MAP: dict[str, str] = {
    # Trình tự thực hiện
    "trình tự thực hiện": "trinh_tu_thuc_hien",
    "trình tự thực hiện:": "trinh_tu_thuc_hien",
    # Cách thức thực hiện
    "cách thức thực hiện": "cach_thuc_thuc_hien",
    "cách thức thực hiện:": "cach_thuc_thuc_hien",
    # Thành phần hồ sơ
    "thành phần hồ sơ": "thanh_phan_ho_so",
    "thành phần hồ sơ:": "thanh_phan_ho_so",
    "thành phần, số lượng hồ sơ": "thanh_phan_ho_so",
    "hồ sơ": "thanh_phan_ho_so",
    # Đối tượng thực hiện
    "đối tượng thực hiện": "doi_tuong_thuc_hien",
    "đối tượng thực hiện:": "doi_tuong_thuc_hien",
    # Cơ quan thực hiện
    "cơ quan thực hiện": "co_quan_thuc_hien",
    "cơ quan thực hiện:": "co_quan_thuc_hien",
    "cơ quan có thẩm quyền": "co_quan_thuc_hien",
    # Kết quả thực hiện
    "kết quả thực hiện": "ket_qua_thuc_hien",
    "kết quả thực hiện:": "ket_qua_thuc_hien",
    "kết quả": "ket_qua_thuc_hien",
    # Căn cứ pháp lý
    "căn cứ pháp lý": "can_cu_phap_ly",
    "căn cứ pháp lý:": "can_cu_phap_ly",
    # Yêu cầu, điều kiện thực hiện
    "yêu cầu, điều kiện thực hiện": "yeu_cau_dieu_kien",
    "yêu cầu, điều kiện thực hiện:": "yeu_cau_dieu_kien",
    "yêu cầu điều kiện thực hiện": "yeu_cau_dieu_kien",
    "điều kiện thực hiện": "yeu_cau_dieu_kien",
    # Thời hạn giải quyết
    "thời hạn giải quyết": "thoi_han_giai_quyet",
    "thời hạn giải quyết:": "thoi_han_giai_quyet",
    "thời hạn": "thoi_han_giai_quyet",
    # Hình thức nộp
    "hình thức nộp": "hinh_thuc_nop",
    "hình thức nộp:": "hinh_thuc_nop",
    "hình thức nộp hồ sơ": "hinh_thuc_nop",
    # Phí, lệ phí
    "phí, lệ phí": "phi_le_phi",
    "phí, lệ phí:": "phi_le_phi",
    "phí lệ phí": "phi_le_phi",
    "lệ phí": "phi_le_phi",
}

# Canonical section keys (ordered by typical document appearance)
KNOWN_SECTIONS = [
    "trinh_tu_thuc_hien",
    "cach_thuc_thuc_hien",
    "thanh_phan_ho_so",
    "doi_tuong_thuc_hien",
    "co_quan_thuc_hien",
    "ket_qua_thuc_hien",
    "can_cu_phap_ly",
    "yeu_cau_dieu_kien",
    "thoi_han_giai_quyet",
    "hinh_thuc_nop",
    "phi_le_phi",
]

# Human-readable display names
SECTION_DISPLAY_NAMES: dict[str, str] = {
    "trinh_tu_thuc_hien": "Trình tự thực hiện",
    "cach_thuc_thuc_hien": "Cách thức thực hiện",
    "thanh_phan_ho_so": "Thành phần hồ sơ",
    "doi_tuong_thuc_hien": "Đối tượng thực hiện",
    "co_quan_thuc_hien": "Cơ quan thực hiện",
    "ket_qua_thuc_hien": "Kết quả thực hiện",
    "can_cu_phap_ly": "Căn cứ pháp lý",
    "yeu_cau_dieu_kien": "Yêu cầu, điều kiện thực hiện",
    "thoi_han_giai_quyet": "Thời hạn giải quyết",
    "hinh_thuc_nop": "Hình thức nộp",
    "phi_le_phi": "Phí, lệ phí",
}
