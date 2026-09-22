import os
import numpy as np
from pypdf import PdfReader

# 引入我們寫好的兩個核心工具 (Stage 1 & Stage 2)
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier

def load_pdf(pdf_path):
    """ 讀取並解析 PDF 檔案 """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def generate_mock_peerread_data(num_samples=100):
    """
    [過渡期函式] 產生虛擬的 PeerRead 訓練資料。
    未來我們可以直接將這裡替換為讀取 PeerRead JSON 的模組。
    """
    print(f"🔄 正在生成 {num_samples} 筆模擬的論文歷史特徵資料...")
    
    # 模擬 768 維的 SciBERT 語意特徵 (-1 到 1 之間)
    semantic_features = np.random.uniform(-1, 1, (num_samples, 768))
    
    # 模擬 5 維統計特徵 [字數, 文獻數, 公式數, 圖表數, Github(0/1)]
    word_counts = np.random.randint(3000, 10000, (num_samples, 1))
    ref_counts = np.random.randint(10, 80, (num_samples, 1))
    eq_counts = np.random.randint(0, 30, (num_samples, 1))
    fig_counts = np.random.randint(0, 15, (num_samples, 1))
    github_links = np.random.randint(0, 2, (num_samples, 1))
    
    statistical_features = np.concatenate(
        (word_counts, ref_counts, eq_counts, fig_counts, github_links), axis=1
    )
    
    # 拼接成 773 維矩陣
    X_mock = np.concatenate((semantic_features, statistical_features), axis=1)
    # 隨機產生錄取結果 (0: Reject, 1: Accept)
    y_mock = np.random.randint(0, 2, num_samples) 
    
    return X_mock, y_mock

def main():
    print("🚀 啟動 autopeer 雙軌制 - SVM 分類快篩管線 (Option A)")
    
    # =========================================================
    # 階段 A: 準備分類器與歷史訓練資料 (Train the SVM)
    # =========================================================
    classifier = ScientificPaperClassifier()
    
    # 生成模擬訓練集與測試集 (暫代 PeerRead)
    X_train, y_train = generate_mock_peerread_data(num_samples=200)
    X_test, y_test = generate_mock_peerread_data(num_samples=50)
    
    # 訓練模型 (這會觸發特徵標準化與 SVM fit)
    classifier.train(X_train, y_train, X_test, y_test)
    
    # =========================================================
    # 階段 B: 對新進 PDF 論文進行特徵抽取 (Feature Extraction)
    # =========================================================
    pdf_path = "data/sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"\n❌ 找不到測試檔案 {pdf_path}")
        return
        
    print(f"\n📄 讀取新進論文 [{pdf_path}] 並進行特徵抽取...")
    raw_text = load_pdf(pdf_path)
    
    # 為了流程順暢，這裡先用寫死的簡短摘要當作錨點
    mock_abstract = "This paper evaluates large language models on automatic paper reviewing tasks."
    
    extractor = ScientificPaperExtractor()
    # 我們只需要 773 維度向量，不需要純文字
    _, hybrid_vector = extractor.extract_core_content(raw_text, mock_abstract, top_k=30)
    
    # =========================================================
    # 階段 C: 預測錄取率與雙閘門判定 (Prediction & Cascade Gate)
    # =========================================================
    print("\n🔮 正在將 773 維特徵送入 SVM 預測錄取機率...")
    pred_class, accept_prob = classifier.predict(hybrid_vector)
    
    print("\n================ 🏆 最終預測結果 ================")
    print(f"➔ 預測結果: {'✅ Accept (錄取)' if pred_class == 1 else '❌ Reject (拒絕)'}")
    print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
    
    # 實作你論文中提到的級聯雙閘門 (Cascade Gate) 概念
    if accept_prob > 0.90 or accept_prob < 0.10:
        print("➔ ⚡ 觸發極端值快篩：免呼叫 LLM，直接以此結果定案！")
    else:
        print("➔ 🔄 落在模糊地帶：建議將論文送入 Option B (大模型生成軌) 進行深度審查。")
    print("=================================================")

if __name__ == "__main__":
    main()