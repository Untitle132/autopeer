import os
import urllib.request
import zipfile

def download_peerread():
    # PeerRead GitHub 倉庫的 ZIP 下載連結
    url = "https://github.com/allenai/PeerRead/archive/refs/heads/master.zip"
    
    # 設定存檔路徑
    data_dir = "data"
    zip_path = os.path.join(data_dir, "PeerRead.zip")
    extract_path = data_dir
    
    # 確保 data 資料夾存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 檢查是否已經下載過，避免重複下載
    if os.path.exists(os.path.join(data_dir, "PeerRead-master")):
        print("✅ PeerRead 資料集已經存在，無需重新下載！")
        return

    print("⏳ 正在從 GitHub 下載 PeerRead 資料集 (約 40MB)，請稍候...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("📥 下載完成！正在解壓縮...")
        
        # 解壓縮 ZIP 檔
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # 清理佔空間的 ZIP 暫存檔
        os.remove(zip_path)
        print(f"🎉 成功！資料集已完整解壓縮至: {os.path.join(data_dir, 'PeerRead-master/')}")
        
    except Exception as e:
        print(f"❌ 下載或解壓縮時發生錯誤: {e}")

if __name__ == "__main__":
    download_peerread()