import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 第一次執行請確保下載了斷句模型
# nltk.download('punkt')

class ScientificPaperExtractor:
    def __init__(self, top_k=30):
        """
        初始化抽取器
        :param top_k: 最終保留給 Generator 的關鍵句子數量
        """
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _split_into_sentences(self, text):
        """ 階段 1: 文本斷句預處理 (Sentence Tokenization) """
        return nltk.sent_tokenize(text)

    def extract_core_content(self, full_text, reference_anchor):
        """
        階段 2: 核心句子抽取演算法
        :param full_text: 論文的全文 (Introduction + Method + Experiments...)
        :param reference_anchor: 核心錨點 (通常傳入 Abstract，因為摘要是全文的精華)
        """
        # 1. 將全文切分為句子陣列
        sentences = self._split_into_sentences(full_text)
        if len(sentences) <= self.top_k:
            return " ".join(sentences) # 若論文太短則無需抽取

        # 2. 特徵提取：將所有句子與錨點（摘要）轉換為 TF-IDF 向量空間
        # 為了計算方便，將摘要加入文本最後一列一起 fit
        all_texts = sentences + [reference_anchor]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        # 3. 分離出句子向量與摘要向量
        sentence_vectors = tfidf_matrix[:-1]
        abstract_vector = tfidf_matrix[-1]

        # 4. 計算學術相關性：計算每個句子與摘要的 Cosine Similarity
        similarity_scores = cosine_similarity(sentence_vectors, abstract_vector).flatten()

        # 5. 排序並選取 Top-K
        # argsort 會由小到大排，我們取最後 top_k 個最大的索引
        top_indices = np.argsort(similarity_scores)[-self.top_k:]
        
        # 關鍵：必須將索引重新排序（由小到大），以確保抽出來的句子「符合論文原本的敘述邏輯與順序」
        top_indices = sorted(top_indices)

        # 6. 重新組合文本
        extracted_sentences = [sentences[i] for i in top_indices]
        return " ".join(extracted_sentences)

# ==========================================
# 研究生測試 Demo (你可以直接執行這段測試)
# ==========================================
if __name__ == "__main__":
    # 模擬從 PDF 讀取出來的長文本
    mock_abstract = "We propose a novel neural network architecture for automated peer review. Our model leverages an extract-then-generate pipeline to handle long documents efficiently."
    
    mock_full_text = (
        "Introduction. Peer review is critical for scientific rigorousness. However, reviewers are overloaded. "
        "In this section, we review related work in text summarization. Many models fail on long inputs. "
        "Methodology. We propose a novel neural network architecture for automated peer review. "
        "Specifically, our model leverages an extract-then-generate pipeline to handle long documents efficiently. "
        "We optimize the framework using a joint multi-task loss function. "
        "Experiments. We evaluate our approach on the PeerRead dataset. The results show significant improvement. "
        "Conclusion. Our method sets a new state-of-the-art for automatic scientific reviewing tasks."
    )

    # 實作抽取器，假設我們只想拿最重要的 3 句話
    extractor = ScientificPaperExtractor(top_k=3)
    compressed_paper = extractor.extract_core_content(mock_full_text, reference_anchor=mock_abstract)

    print("=== 論文原本總字數 ===")
    print(len(mock_full_text.split()))
    print("\n=== 抽取後的精華文本 (餵給 Generator 的輸入) ===")
    print(compressed_paper)