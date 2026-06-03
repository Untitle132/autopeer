import nltk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class ScientificPaperExtractor:
    def __init__(self, top_k=30):
        """
        初始化基於 SciBERT 的語意抽取器
        :param top_k: 最終保留給 Generator 的關鍵句子數量
        """
        self.top_k = top_k
        
        # 使用專門處理科學文獻的 SciBERT 模型
        self.model_name = "allenai/scibert_scivocab_uncased"
        print(f"⏳ 正在載入本地端 {self.model_name} 模型與 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # 檢查是否有 GPU 加速 (在研究生實驗室環境特別重要)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval() # 切換至推論模式
        print(f"✅ 模型載入完成。當前運行裝置: {self.device}")

    def _split_into_sentences(self, text):
        """ 階段 1: 文本斷句預處理 """
        return nltk.sent_tokenize(text)

    def _get_embeddings(self, texts):
        """
        利用 SciBERT 提取文本的 Dense Embedding
        """
        embeddings = []
        # 為了避免 OOM (記憶體溢出)，採用 Batch 處理或逐句處理
        with torch.no_grad():
            for text in texts:
                # 進行 Tokenization，並限制最大長度防止超出 BERT 的 512 限制
                inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # 取出 [CLS] Token 的向量作為整句話的語意代表 (Shape: [1, 768])
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
                
        return np.array(embeddings)

    def extract_core_content(self, full_text, reference_anchor):
        """
        階段 2: 核心句子抽取演算法 (升級為 SciBERT 語意向量 + Cosine Similarity)
        :param full_text: 論文的全文
        :param reference_anchor: 核心錨點 (通常傳入 Abstract)
        """
        sentences = self._split_into_sentences(full_text)
        if len(sentences) <= self.top_k:
            return " ".join(sentences)

        # 1. 提取所有句子與 Abstract 的 SciBERT 向量
        # 為了加速，我們分開處理：前面是論文各句，最後一個是 Abstract
        sentence_embeddings = self._get_embeddings(sentences) # Shape: [N, 768]
        abstract_embedding = self._get_embeddings([reference_anchor]) # Shape: [1, 768]

        # 2. 計算所有句子向量與 Abstract 向量之間的 Cosine Similarity
        similarity_scores = cosine_similarity(sentence_embeddings, abstract_embedding).flatten()

        # 3. 根據分數排序，取得前 Top-K 個句子的索引值
        top_indices = np.argsort(similarity_scores)[-self.top_k:]
        top_indices = sorted(top_indices) # 恢復句子在原文中的先後順序，確保 Context 的邏輯連貫

        # 4. 重組文本
        extracted_sentences = [sentences[i] for i in top_indices]
        return " ".join(extracted_sentences)