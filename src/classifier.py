import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class ScientificPaperClassifier:
    def __init__(self):
        # 初始化 SVM 模型，啟用 probability 以便輸出錄取機率
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        # 初始化標準化工具
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        訓練 SVM 分類器
        X_train: Shape (n_samples, 773) 的 Numpy 陣列
        y_train: Shape (n_samples,) 的標籤陣列 (1: Accept, 0: Reject)
        """
        print("⚙️ 正在進行特徵尺度對齊 (Standard Scaling)...")
        # 訓練 StandardScaler 並轉換訓練資料
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("🧠 正在訓練 SVM 模型...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        print("✅ 模型訓練完成！")

        # 如果有提供測試集，則順便印出評估報告
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """評估模型效能並印出報告"""
        if not self.is_trained:
            print("⚠️ 模型尚未訓練，無法評估！")
            return

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n================ 📊 SVM 快篩分類器效能報告 ================")
        print(f"➔ 準確率 (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("\n詳細分類指標:")
        print(classification_report(y_test, y_pred, target_names=['Reject (0)', 'Accept (1)']))
        print("===========================================================\n")

    def predict(self, X):
        """
        對新論文進行錄取預測
        回傳: (預測類別 [0或1], 錄取機率 [0.0 ~ 1.0])
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先呼叫 train() 或 load_model()。")
            
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        # predict_proba 回傳 [[Reject機率, Accept機率]]
        accept_probability = self.model.predict_proba(X_scaled)[0][1] 
        
        return prediction, accept_probability

    def save_model(self, save_dir="models"):
        """將訓練好的模型與 Scaler 儲存至本機"""
        if not self.is_trained:
            print("⚠️ 模型尚未訓練，沒有東西可以儲存！")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_dir, 'svm_model.pkl'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        print(f"💾 模型已成功儲存至 {save_dir}/ 資料夾中。")

    def load_model(self, save_dir="models"):
        """從本機載入已訓練的模型與 Scaler"""
        model_path = os.path.join(save_dir, 'svm_model.pkl')
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print("📂 成功載入已訓練的 SVM 模型與 Scaler。")
        else:
            print(f"❌ 找不到模型檔案，請確認 {save_dir}/ 目錄下是否存在 pkl 檔。")