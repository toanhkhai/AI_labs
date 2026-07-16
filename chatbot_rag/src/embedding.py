import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Đọc file jsonl (các chunking)
def read_chunks(path):
    chunks = []
    with open (path,'r',encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Đã đọc {len(chunks)} các đoạn từ {path}")
    return chunks

# Tạo các embeddings từ các chunks
def create_embedding(chunks,model):
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk['chunk_text'])
    embeddings = model.encode(chunk_texts)
    return embeddings

# Tạo các metadata cho các chunk
def save_metadata(chunks):
    metadata = []
    for chunk in chunks:
        metadata.append({
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id']
        })
    return metadata

if __name__ == "__main__":

    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2",device=0)

    chunks_token = read_chunks("build_a_semantic_search_pipeline/chunks/chunks_overlap.jsonl")
    chunks_sentence = read_chunks("build_a_semantic_search_pipeline/chunks/chunks_sentences.jsonl")

    #Tạo embedding cho cách chunk token.
    embeddings_chunkToken = create_embedding(chunks_token,model)
    metadata_chunkToken = save_metadata(chunks_token)
    print(f"Đã tạo ma trận embedding và meta theo cách chunk_token có overlap.Kích thước embedding: {embeddings_chunkToken.shape}")

    #Tạo embedding cho cách chunk sentence.
    embeddings_chunkSentence = create_embedding(chunks_sentence,model)
    metadata_chunkSentence = save_metadata(chunks_sentence)
    print(f"Đã tạo ma trận embedding và metadata theo cách chunk_sentence. Kích thước embedding: {embeddings_chunkSentence.shape}")

    import chromadb
    chromadb_client = chromadb.PersistentClient("build_a_semantic_search_pipeline/vector_database/ChromaDB/")

    collection_token = chromadb_client.get_or_create_collection(name="token_overlap_collection")
    collection_sentence = chromadb_client.get_or_create_collection(name="sentence_collection")

    collection_token.upsert(
        ids= [ chunk['chunk_id'] for chunk in chunks_token],
        documents=[ chunk['chunk_text'] for chunk in chunks_token ],
        metadatas= metadata_chunkToken,
        embeddings= embeddings_chunkToken.tolist()
    )

    collection_sentence.upsert(
        ids= [ chunk['chunk_id'] for chunk in chunks_sentence ],
        documents= [ chunk['chunk_text'] for chunk in chunks_sentence ],
        metadatas= metadata_chunkSentence,
        embeddings=embeddings_chunkSentence.tolist()
    )
    
    print("Đã lưu vector và chromaDB")