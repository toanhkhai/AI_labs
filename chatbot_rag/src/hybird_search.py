from sentence_transformers import SentenceTransformer
import chromadb
import json
import rank_bm25
from flashrank import Ranker, RerankRequest


#================VECTOR SEARCH=================#
#Khởi tạo embedding model
model_kwargs = {"attn_implementation": "sdpa"}
model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2",device=0,model_kwargs=model_kwargs)

#Kết nối đến vector database
chroma_client = chromadb.PersistentClient("build_a_semantic_search_pipeline/vector_database/ChromaDB/")
collection = chroma_client.get_collection(name="sentence_collection")

def vector_search(message_query,top_k):
    #Câu query của người dùng
    query = message_query
    query_vector = model.encode(query).tolist()

    #Kết quả query từ database 
    result = collection.query(
        query_embeddings=[query_vector],
        n_results= top_k,
        include=["documents","metadatas"]
    )
    if result and result.get('documents') and len(result['documents']) > 0:
        return result['documents'][0]
    return []


#==================KEY SEARCH==================#
#Đọc file chunks 
path = "build_a_semantic_search_pipeline/chunks/chunks_sentences.jsonl"
chunks = []
with open (path,'r',encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

documents = [ chunk['chunk_text'] for chunk in chunks]
tokenized_docs = [ doc.lower().split(" ") for doc in documents]

#Khởi tạo bm25, tính điểm 
bm25 = rank_bm25.BM25Okapi(tokenized_docs)

def key_search(message_query, top_k):
    tokenized_query = message_query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    sorted_indices = sorted(range(len(bm25_scores)), key=lambda k: bm25_scores[k], reverse=True)
    top_indices = sorted_indices[:top_k]

    key_search_documents = [documents[index] for index in top_indices]
    return key_search_documents


#================HYBRID SEARCH==============#
ranker = Ranker()
def hybrid_search(message_query, top_k_retrieve=10,top_k_final=3):
    documents_vector_search = vector_search(message_query,top_k_retrieve)
    documents_key_search = key_search(message_query,top_k_retrieve)

    all_documents = []
    for text in documents_vector_search:
        if text not in all_documents:
            all_documents.append(text)

    for text in documents_key_search:
        if text not in all_documents:
            all_documents.append(text)

    passages_for_rerank = []
    for index, text in enumerate(all_documents):
        passages_for_rerank.append(
            {
                "id": index,  # Cần một ID số bất kỳ để định danh ứng viên
                "text": text,  # Chuỗi văn bản thô để mô hình đọc hiểu
            }
        )
    
    re_rank_request = RerankRequest(query=message_query,passages=passages_for_rerank)
    re_rank_result = ranker.rerank(re_rank_request)
    top_k_result = re_rank_result[:top_k_final]
    final_result = [item['text'] for item in top_k_result]
    return final_result


if __name__ == "__main__":
    message_query = "Đại học bách khoa Hà Nội đứng thứ mấy trong bảng xếp hạng"

    # Lấy diện rộng mỗi bên 10 ứng viên, sàng lọc và rerank giữ lại đúng top 3 chuỗi text hay nhất
    final_results = hybrid_search(
        message_query, top_k_retrieve=10, top_k_final=5
    )

    print(f"--- KẾT QUẢ DANH SÁCH HYBRID SEARCH (TOP 3) ---\n")
    for index, chunk_text in enumerate(final_results):
        print(f"Hạng {index + 1}:")
        print(f"{chunk_text}")
        print("-" * 80)
