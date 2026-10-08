import torch  # 🌟 務必放在第一行：強制搶先載入核心，避免 Windows DLL (WinError 1114) 報錯

import os
import numpy as np
from pypdf import PdfReader
from src.dataset_loader import PeerReadLoader
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.llm_reviewer import LLMReviewer

# 💡 升級版適配器：動態偵測回傳數量，並自動補齊 Gemini 需要的文本
def extract_auto(extractor_obj, text):
    result = extractor_obj._extract_statistical_features(text)
    
    if isinstance(result, tuple):
        if len(result) >= 3:
            return result[0], result[1], result[2]
        elif len(result) == 2:
            return result[0], None, result[1]
        else:
            return result[0], None, text[:1000]
    else:
        fallback_sentences = text[:1000] if isinstance(text, str) else "無法取得內文"
        return result, None, fallback_sentences

def main():
    print("🚀 啟動 AutoPeer 雙軌制自動審查系統...")
    
    loader = PeerReadLoader()
    extractor = ScientificPaperExtractor()
    classifier = ScientificPaperClassifier()
    llm_reviewer = LLMReviewer()

    papers = loader.load_data()
    if not papers:
        print("❌ 找不到訓練資料，系統中止。")
        return
        
    train_papers = papers[:30] 
    print(f"\n⏳ 正在使用 SciBERT 抽取 {len(train_papers)} 篇訓練論文的特徵...")
    
    # ⚠️ 這裡的空列表是用來收集資料的，絕對不能被刪掉
    X_train = []
    y_train = []
    
    for i, paper in enumerate(train_papers):
        if (i + 1) % 5 == 0:
            print(f"   ... 已處理 {i + 1}/{len(train_papers)} 篇")
        
        vec, _, _ = extract_auto(extractor, paper["text"])
        X_train.append(vec)
        y_train.append(paper["label"])
        
    # ✨ 降維修復：強制將 3D 陣列壓扁為 2D 表格
    X_train = np.array(X_train).reshape(len(train_papers), -1)
    y_train = np.array(y_train)

    print("\n⚙️ 正在訓練 SVM 分類器...")
    classifier.train(X_train, y_train, X_train, y_train)

    sample_pdf_path = "data/sample_paper.pdf"
    print(f"\n🔍 正在分析目標論文: {sample_pdf_path} ...")
    
    try:
        reader = PdfReader(sample_pdf_path)
        test_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ 讀取 {sample_pdf_path} 失敗: {e}")
        return

    # 抽取目標論文的特徵與關鍵句
    test_vector, _, top_sentences = extract_auto(extractor, test_text)
    
    # ✨ 強制將 1D 陣列轉為 1 列的 2D 矩陣
    test_vector = np.array(test_vector).reshape(1, -1)
    
    pred_class, accept_prob = classifier.predict(test_vector)
    
    print("\n================ 🏆 Stage 1: SVM 快篩結果 ================")
    print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
    
    if 0.20 <= accept_prob <= 0.80:
        print("⚠️ 錄取率落在模糊地帶，觸發 Stage 2 (Gemini 深度審查)...")
        print("========================================================\n")
        
        print("⏳ 正在呼叫 Gemini 生成結構化深度評論...")
        review_result = llm_reviewer.generate_review(top_sentences)
        
        print("\n================ 🧠 Stage 2: Gemini 審查報告 ================")
        print(review_result)
        print("=============================================================")
    else:
        decision = "✅ 直接錄取" if accept_prob > 0.80 else "❌ 直接拒絕"
        print(f"➔ 決策結果: {decision} (分數落於極端值，無需動用 LLM 審查)")
        print("========================================================\n")

if __name__ == "__main__":
    main()