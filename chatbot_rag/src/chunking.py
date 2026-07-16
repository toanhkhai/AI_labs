import os
import json
from nltk.tokenize import sent_tokenize
import tiktoken

# Khởi tạo bộ mã hóa văn bản thành các token.
tokenization = tiktoken.get_encoding("cl100k_base")


# Đọc data.
def read_documents(thu_muc="build_a_semantic_search_pipeline/data/"):
    """Đọc toàn bộ file .txt trong thư mục"""
    tai_lieu = []
    for file in os.listdir(thu_muc):
        if file.endswith(".txt"):
            path = os.path.join(thu_muc, file)
            with open(path, 'r', encoding="utf-8") as f:
                tai_lieu.append({
                    'id': file.replace('.txt', ''),
                    'name': file,
                    'content': f.read()
                })
    # Trả về danh sách các dic, mỗi một dic là một file tài liệu.
    return tai_lieu



#Chia theo token cố định áp dụng kỹ thuật overlap 
#Mỗi khúc cắt ra thì nhích lên trên 40 token và cắt đúng 250 token, không tăng thêm thành 290
#Mỗi chunk phải có cấu trúc dic
class ChunkingTokenOverlap:
    def __init__(self, chunk_size=250, overlap=40):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunking(self, ma_tai_lieu, tai_lieu):
        if self.overlap >= self.chunk_size:
            raise ValueError("Kích thước overlap không được lớn hơn hoặc bằng kích thước chunk token!")
        chunks = []
        tokens = tokenization.encode(tai_lieu)
        token_dang_xet = 0  
        so_thu_tu = 1
        while token_dang_xet < len(tokens):
            token_ket_thuc = min(token_dang_xet + self.chunk_size, len(tokens))
            doan_van_ban = tokenization.decode(tokens[token_dang_xet:token_ket_thuc])
            chunks.append({
                'chunk_id': f"{ma_tai_lieu}_token_overlap_{so_thu_tu}",
                'doc_id': ma_tai_lieu,
                'chunk_text': doan_van_ban.strip(),
                'start_position': token_dang_xet,
                'end_position': token_ket_thuc
            })
            token_dang_xet = token_dang_xet + (self.chunk_size - self.overlap)
            so_thu_tu += 1 
        return chunks 



#Dùng thư viện nlkt để cắt một đoạn thành nhiều câu. Giới hạn 1 lần cắt tối đa 300 token và 5 câu,cắt tối thiểu 3 câu.
#Ban đâu dùng nlkt cắt từng thành từng câu một. Ghép các câu lại thành 1 đoạn (một chunk),khi nào vi phạm điều kiện thì cắt
class ChunkingSentence:
    """Chia văn bản dựa theo các câu hoàn chỉnh và giới hạn số token"""
    def __init__(self, token_toi_da=300, so_cau_toi_thieu=3, so_cau_toi_da=5):
        self.token_toi_da = token_toi_da
        self.so_cau_toi_thieu = so_cau_toi_thieu
        self.so_cau_toi_da = so_cau_toi_da

    def chunking(self, ma_tai_lieu, tai_lieu):
        cac_cau = sent_tokenize(tai_lieu)
        cac_doan_chunk = []
        hop_chua_cau = []
        so_token_trong_hop = 0
        so_thu_tu_doan = 1 # Đổi thành bắt đầu từ 1 cho đồng bộ với Chunker cố định
        for cau in cac_cau:
            so_token_cau_hien_tai = len(tokenization.encode(cau))
            # ĐIỀU KIỆN 1: Kiểm tra TRƯỚC KHI NẠP câu mới vào hộp
            # Nếu thêm câu mới bị nổ 300 token VÀ hộp hiện tại đã có đủ số câu tối thiểu -> Cắt hộp cũ
            du_kien_token = so_token_trong_hop + so_token_cau_hien_tai
            hop_du_so_cau_toi_thieu = len(hop_chua_cau) >= self.so_cau_toi_thieu
            if (du_kien_token > self.token_toi_da) and hop_du_so_cau_toi_thieu:
                cac_doan_chunk.append({
                    'chunk_id': f"{ma_tai_lieu}_sentence_{so_thu_tu_doan}",
                    'doc_id': ma_tai_lieu,
                    'so_thu_tu': so_thu_tu_doan,
                    'chunk_text': ' '.join(hop_chua_cau).strip(),
                    'so_cau': len(hop_chua_cau),
                    'so_token_xap_xi': so_token_trong_hop,
                    'cach_chia': 'sentence'
                })
                # Dọn dẹp hộp trống để chuẩn bị cho lượt lưu tiếp theo
                hop_chua_cau, so_token_trong_hop = [], 0
                so_thu_tu_doan += 1

            # Nạp câu đang xét vào hộp
            hop_chua_cau.append(cau)
            so_token_trong_hop += so_token_cau_hien_tai
            
            # ĐIỀU KIỆN 2: Kiểm tra SAU KHI NẠP
            # Hộp đầy ắp câu (Đạt số câu tối đa cho phép) -> Ép buộc cắt ngay lập tức bất kể token
            if len(hop_chua_cau) >= self.so_cau_toi_da:
                cac_doan_chunk.append({
                    'chunk_id': f"{ma_tai_lieu}_sentence_{so_thu_tu_doan}",
                    'doc_id': ma_tai_lieu,
                    'so_thu_tu': so_thu_tu_doan,
                    'chunk_text': ' '.join(hop_chua_cau).strip(),
                    'so_cau': len(hop_chua_cau),
                    'so_token_xap_xi': so_token_trong_hop,
                    'cach_chia': 'sentence'
                })
                # Reset bộ đếm
                hop_chua_cau, so_token_trong_hop = [], 0
                so_thu_tu_doan += 1

        # Đóng gói nốt những câu còn sót lại cuối cùng sau khi hết vòng lặp
        if hop_chua_cau:
            cac_doan_chunk.append({
                'chunk_id': f"{ma_tai_lieu}_sentence_{so_thu_tu_doan}",
                'doc_id': ma_tai_lieu,
                'so_thu_tu': so_thu_tu_doan,
                'chunk_text': ' '.join(hop_chua_cau).strip(),
                'so_cau': len(hop_chua_cau),
                'so_token_xap_xi': so_token_trong_hop,
                'cach_chia': 'sentence'
            })
            
        return cac_doan_chunk

# Lưu các chunk thành file JSONL
def save_chunking(cac_doan, duong_dan):
    os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
    with open(duong_dan, 'w', encoding='utf-8') as f:
        for doan in cac_doan:
            f.write(json.dumps(doan, ensure_ascii=False) + '\n')
    print(f"Đã lưu {len(cac_doan)} đoạn vào {duong_dan}")



if __name__ == "__main__":
    
    list_documents = read_documents()   

    all_chunks_token = []
    all_chunks_sentence = []

    chunking_token_overlap = ChunkingTokenOverlap()
    chunking_sentence = ChunkingSentence()
    
    for docs in list_documents:
        print(f"\nĐang xử lý: {docs['id']}")
        
        # 1. Chia theo kích thước token cố định có dùng overlap
        chunks_token = chunking_token_overlap.chunking(docs['id'], docs['content'])
        all_chunks_token.extend(chunks_token)
        print(f"  Cố định: {len(chunks_token)} đoạn")
        
        # 2. Chia theo câu, gôm lại thành 1 đoạn.
        chunks_sentence = chunking_sentence.chunking(docs['id'], docs['content'])
        all_chunks_sentence.extend(chunks_sentence)
        print(f"  Theo câu: {len(chunks_sentence)} đoạn\n")
    
    # Lưu kết quả đầu ra
    save_chunking(all_chunks_token, "build_a_semantic_search_pipeline/chunks/chunks_overlap.jsonl")
    save_chunking(all_chunks_sentence,"build_a_semantic_search_pipeline/chunks/chunks_sentences.jsonl")
    
    print(f"\nTỔNG KẾT:\n  Đoạn cố định: {len(all_chunks_token)}\n  Đoạn theo câu: {len(all_chunks_sentence)}")