import os
import numpy as np
from pypdf import PdfReader

# 引入核心工具
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.dataset_loader import PeerReadLoader  # 新增的 Data Loader

def load_pdf(pdf_path):
    """ 讀取並解析測試用 PDF 檔案 """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def prepare_real_training_data(extractor, num_samples=30):
    """ 呼叫 Loader 讀取真實論文，並透過 Extractor 轉換為 773 維特徵 """
    loader = PeerReadLoader()
    raw_texts, abstracts, labels = loader.load_dataset(max_samples=num_samples)
    
    feature_list = []
    print(f"\n⏳ 正在透過 SciBERT 抽取 {len(labels)} 篇論文的 773 維特徵...")
    print("   (本機執行需一點時間，請耐心等候 ☕)")
    
    for i, (text, abstract) in enumerate(zip(raw_texts, abstracts)):
        if i % 5 == 0 and i > 0:
            print(f"  - 處理進度: {i}/{len(labels)}")
            
        # 將真實論文餵給你的 Extractor
        _, hybrid_vector = extractor.extract_core_content(text, abstract, top_k=30)
        feature_list.append(hybrid_vector[0])
        
    return np.array(feature_list), labels

def main():
    print("🚀 啟動 autopeer 雙軌制 - 真實數據 SVM 快篩管線")
    
    # =========================================================
    # 階段 A: 準備分類器與真實訓練資料
    # =========================================================
    extractor = ScientificPaperExtractor()
    classifier = ScientificPaperClassifier()
    
    # 載入並抽取 30 篇真實論文特徵 (設定 30 篇是為了讓筆電能在 1-2 分鐘內跑完)
    X_data, y_data = prepare_real_training_data(extractor, num_samples=30)
    
    # 切割訓練集 (80%) 與測試集 (20%)
    split_idx = int(len(X_data) * 0.8)
    X_train, y_train = X_data[:split_idx], y_data[:split_idx]
    X_test, y_test = X_data[split_idx:], y_data[split_idx:]
    
    classifier.train(X_train, y_train, X_test, y_test)
    
    # =========================================================
    # 階段 B: 對新進 PDF 論文進行特徵抽取
    # =========================================================
    pdf_path = "data/sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"\n❌ 找不到測試檔案 {pdf_path}")
        return
        
    print(f"\n📄 讀取新進論文 [{pdf_path}] 並進行特徵抽取...")
    raw_text = load_pdf(pdf_path)
    mock_abstract = "This paper evaluates large language models on automatic paper reviewing tasks."
    
    _, hybrid_vector = extractor.extract_core_content(raw_text, mock_abstract, top_k=30)
    
    # =========================================================
    # 階段 C: 預測錄取率與雙閘門判定
    # =========================================================
    print("\n🔮 正在將 773 維特徵送入 SVM 預測錄取機率...")
    pred_class, accept_prob = classifier.predict(hybrid_vector)
    
    print("\n================ 🏆 最終預測結果 ================")
    print(f"➔ 預測結果: {'✅ Accept (錄取)' if pred_class == 1 else '❌ Reject (拒絕)'}")
    print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
    
    if accept_prob > 0.80 or accept_prob < 0.20:
        print("➔ ⚡ 觸發極端值快篩：免呼叫 LLM，直接以此結果定案！")
    else:
        print("➔ 🔄 落在模糊地帶：建議將論文送入 Option B (大模型生成軌) 進行深度審查。")
    print("=================================================")

if __name__ == "__main__":
    main()