import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from sklearn.metrics.pairwise import cosine_similarity

class ScientificPaperExtractor:
    def __init__(self):
        # 載入針對科學文獻預訓練的 SciBERT 模型與 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        # 自動偵測是否有 GPU (CUDA) 加速
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval() # 設為評估模式

    def _get_scibert_embedding(self, text):
        """將單一文本轉換為 768 維的 SciBERT 密集向量"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 取 [CLS] token 的隱藏層輸出作為句子特徵向量 (Shape: [1, 768])
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embedding

    def _extract_statistical_features(self, raw_text):
        """
        右通道：提取 5 維淺層統計特徵 (暴力破解與除錯版)
        """
        # 1. 總字數
        word_count = len(raw_text.split())
        
        # 2. 參考文獻數：暴力抓取 'et al' (無視標點符號與年份) 與 [數字]
        # pypdf 常常會把 'et al.' 變成 'et al .' 甚至斷行
        ref_matches_ieee = re.findall(r'\[\s*\d+\s*\]', raw_text)
        ref_matches_apa = re.findall(r'(?i)et\s*al', raw_text)
        ref_count = len(ref_matches_ieee) + len(ref_matches_apa)
        
        # 3. 公式數量：暴力抓取 Eq 或 Equation 後面跟數字
        eq_matches = re.findall(r'(?i)\bEq(?:uation|\.)?[\s\n]*\(?\d+\)?', raw_text)
        eq_count = len(eq_matches)
        
        # 4. 圖表總數：維持原樣 (這個你測出來 14 是準確的 Proxy 特徵)
        fig_table_count = len(re.findall(r'(?i)\b(?:Fig\.|Figure|Table)[\s\n]+\d+', raw_text))
        
        # 5. 開源連結：只要字串內出現 github 或 huggingface 就給 1
        link_matches = re.findall(r'(?i)github|huggingface|zenodo', raw_text)
        has_github = 1.0 if len(link_matches) > 0 else 0.0
        
        # ---- [Debug 專用] 印出到底抓到了什麼字串 ----
        print("\n--- 🕵️‍♂️ Regex 抓取除錯報告 ---")
        print(f"文獻 (APA et al): 抓到 {len(ref_matches_apa)} 次 -> 預覽: {ref_matches_apa[:5]}")
        print(f"公式 (Eq.): 抓到 {len(eq_matches)} 次 -> 預覽: {eq_matches[:5]}")
        print(f"開源連結: 抓到 {len(link_matches)} 次 -> 預覽: {link_matches}")
        print("------------------------------\n")
        
        return np.array([[word_count, ref_count, eq_count, fig_table_count, has_github]], dtype=np.float32)

    def extract_core_content(self, raw_text, abstract_text, top_k=30):
        """
        雙軌特徵融合主程式碼
        回傳： (Top-K精華純文字, 773維混合特徵向量)
        """
        # 1. 斷句預處理
        sentences = nltk.sent_tokenize(raw_text)
        if not sentences:
            return "", np.zeros((1, 773))

        # 2. 取得 Abstract 的錨點向量 (Shape: [1, 768])
        abstract_embedding = self._get_scibert_embedding(abstract_text)

        # 3. 取得所有句子的向量並計算相似度
        sentence_embeddings = []
        similarities = []
        
        for sentence in sentences:
            # 過濾掉太短的無意義雜訊
            if len(sentence.split()) < 5:
                continue
                
            sent_emb = self._get_scibert_embedding(sentence)
            sentence_embeddings.append(sent_emb)
            
            # 計算與 Abstract 的 Cosine Similarity
            sim = cosine_similarity(sent_emb, abstract_embedding)[0][0]
            similarities.append(sim)

        # 防呆機制：如果有效句子少於 top_k
        actual_k = min(top_k, len(similarities))
        
        # 4. 挑出分數最高的 Top-K 句子的 Index
        top_indices = np.argsort(similarities)[-actual_k:]
        
        # 5. 依據原文順序重新排序 (恢復時序)
        top_indices_sorted = sorted(top_indices)
        
        # 組合 Context 給 LLM (生成軌道用)
        extracted_sentences = [sentences[i] for i in top_indices_sorted]
        extracted_text = " ".join(extracted_sentences)

        # --- 組合特徵給 Classifier (快篩軌道用) ---
        
        # [左通道]：將精選出的句子向量進行 Mean Pooling (平均池化)，獲得代表整篇的 [1, 768] 向量
        top_embeddings = [sentence_embeddings[i] for i in top_indices_sorted]
        semantic_vector = np.mean(top_embeddings, axis=0) # Shape: [1, 768]
        
        # [右通道]：呼叫 Regex 抓取 5 維統計特徵
        statistical_vector = self._extract_statistical_features(raw_text) # Shape: [1, 5]
        
        # [多源特徵融合]：利用 np.concatenate 拼接成 [1, 773] 的混合特徵向量
        hybrid_feature_vector = np.concatenate((semantic_vector, statistical_vector), axis=1)

        return extracted_text, hybrid_feature_vector
