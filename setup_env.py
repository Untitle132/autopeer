import subprocess
import sys

def setup_environment():
    # AutoPeer 系統依賴套件清單
    packages = [
        "numpy",
        "scikit-learn",       # 支援下游任務的 StandardScaler 特徵正規化與 SVM 分類器
        "pypdf",              # 負責讀取並提取科學 PDF 的純文字與頁面 
        "nltk",               # 處理特徵抽取階段的 Tokenization 
        "torch",              # SciBERT 語意向量運算的底層框架
        "transformers",       # 載入 768 維度的 SciBERT Dense Vector 模型 
        "google-generativeai" # Option B 文本生成階段的 Gemini API SDK
    ]

    print("🚀 啟動 AutoPeer 環境建置程序...")
    
    # 自動安裝所有 Python 套件
    for package in packages:
        print(f"📦 正在安裝 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("\n📚 正在下載 NLTK 專用的斷句語料庫...")
    import nltk
    # 下載 Punkt 模型以支援精準的句子切割
    nltk.download('punkt')
    # 針對新版 NLTK 的補充語料庫
    nltk.download('punkt_tab') 

    print("\n🎉 所有環境建置完成！您可以開始執行 main_classify.py 了。")

if __name__ == "__main__":
    setup_environment()