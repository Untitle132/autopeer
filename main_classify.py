import torch  # 🌟 務必放在第一行：強制搶先載入核心，避免 Windows DLL (WinError 1114) 報錯

import os
import numpy as np
from pypdf import PdfReader
from src.dataset_loader import PeerReadLoader
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.llm_reviewer import LLMReviewer

def main():
    print("🚀 啟動 AutoPeer 雙軌制自動審查系統...")
    
    # ==========================================
    # 1. 模組初始化
    # ==========================================
    loader = PeerReadLoader()
    extractor = ScientificPaperExtractor()
    classifier = ScientificPaperClassifier()
    llm_reviewer = LLMReviewer()

    # ==========================================
    # 2. 載入並萃取 PeerRead 真實資料集
    # ==========================================
    papers = loader.load_data()
    if not papers:
        print("❌ 找不到訓練資料，請確認 data/ 目錄下是否有解壓縮的 JSON 檔案。系統中止。")
        return
        
    # 測試階段，我們先取前 30 篇來做快速訓練驗證
    train_papers = papers[:30] 
    print(f"\n⏳ 正在使用 SciBERT 抽取 {len(train_papers)} 篇訓練論文的 773 維特徵...")
    
    X_train = []
    y_train = []
    for i, paper in enumerate(train_papers):
        if (i + 1) % 5 == 0:
            print(f"   ... 已處理 {i + 1}/{len(train_papers)} 篇")
        
        # 萃取特徵向量
        vec, _, _ = extractor.extract_features(paper["text"])
        X_train.append(vec)
        y_train.append(paper["label"])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # ==========================================
    # 3. 訓練 SVM 分類器 (Stage 1)
    # ==========================================
    print("\n⚙️ 正在訓練 SVM 分類器...")
    # 在工程測試階段，我們先將測試集設與訓練集相同以驗證管線暢通
    classifier.train(X_train, y_train, X_train, y_train)

    # ==========================================
    # 4. 分析目標論文 (sample_paper.pdf)
    # ==========================================
    sample_pdf_path = "data/sample_paper.pdf"
    print(f"\n🔍 正在分析目標論文: {sample_pdf_path} ...")
    
    try:
        # 讀取 PDF 純文字
        reader = PdfReader(sample_pdf_path)
        test_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ 讀取 {sample_pdf_path} 失敗: {e}")
        return

    # 抽取目標論文的特徵與關鍵句
    test_vector, _, top_sentences = extractor.extract_features(test_text)
    
    # ✨ 關鍵修復：強制將 1D 陣列轉為 1 列、773 欄的 2D 矩陣，避免 SVM 報錯
    test_vector = np.array(test_vector).reshape(1, -1)
    
    # ==========================================
    # 5. 雙軌制智能路由 (SVM 快篩 -> 條件式 LLM)
    # ==========================================
    pred_class, accept_prob = classifier.predict(test_vector)
    
    print("\n================ 🏆 Stage 1: SVM 快篩結果 ================")
    print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
    
    # 定義模糊地帶 (例如 20% ~ 80%)
    if 0.20 <= accept_prob <= 0.80:
        print("⚠️ 錄取率落在模糊地帶，觸發 Stage 2 (Gemini 深度審查)...")
        print("========================================================\n")
        
        # 將 Extractor 抓出的 Top-K 關鍵句送給 Gemini
        print("⏳ 正在呼叫 Gemini 生成結構化深度評論...")
        review_result = llm_reviewer.generate_review(top_sentences)
        
        print("\n================ 🧠 Stage 2: Gemini 審查報告 ================")
        print(review_result)
        print("=============================================================")
    else:
        # 若分數極端 (低於 20% 或高於 80%)，直接定案，節省 API 成本
        decision = "✅ 直接錄取" if accept_prob > 0.80 else "❌ 直接拒絕"
        print(f"➔ 決策結果: {decision} (分數落於極端值，無需動用 LLM 審查)")
        print("========================================================\n")

if __name__ == "__main__":
    main()