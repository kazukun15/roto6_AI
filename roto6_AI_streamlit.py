import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import requests

# --- 基本設定 ---
st.set_page_config(page_title="ロト6データ分析アプリ", layout="wide")
st.title("ロト6データ分析アプリ")
st.markdown("### 過去の抽選結果データから次回のロト6の予測番号を出力します")

# oneDNN の最適化を無効化（必要に応じて）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# --- サイドバー設定 ---
st.sidebar.header("【設定】")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Gemini APIキーを入力してください。")
st.sidebar.markdown("※ 本番環境ではキーは安全に管理してください")
analysis_method = st.sidebar.radio(
    "分析方法を選択",
    ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク", "Gemini API")
)

# --- CSV読み込み＆前処理関数 ---
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding="utf-8")
    return df

# --- 機械学習用関数群 ---
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    n_classes = y_encoded.shape[1]
    return X_scaled, y_encoded, n_classes

def build_nn_model(input_dim, units, dropout, learning_rate, n_classes):
    model = Sequential([
        Dense(units, activation='relu', input_dim=input_dim),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def optimize_hyperparameters(X_train, y_train, n_classes):
    def objective(trial):
        units = trial.suggest_int('units', 32, 256, step=32)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 10, 50, step=10)
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
        model = build_nn_model(
            input_dim=X_train.shape[1],
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            n_classes=n_classes
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        _, accuracy = model.evaluate(X_train, y_train, verbose=0)
        return accuracy
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

# --- Gemini API 呼び出し関数 (プロンプト改善版) ---
def get_gemini_predictions(api_key, prompt_text):
    # APIキーはURLのクエリパラメータで渡す
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json().get('predictions', [])
        return predictions
    except requests.exceptions.RequestException as e:
        st.error(f"Gemini APIエラー: {e}")
        return []

# --- プログレスバー更新用コールバック ---
class ProgressBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs
        self.current_epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress_rate = int(100 * self.current_epoch / self.total_epochs)
        self.progress_bar.progress(progress_rate)

# --- 結果出力用関数 ---
def print_predicted_numbers_top6(prob_array, n=5):
    st.write("#### 予想される数字（各サンプルの上位6クラス）")
    for i in range(min(n, prob_array.shape[0])):
        sorted_indices = np.argsort(prob_array[i])[::-1]
        top6 = sorted_indices[:6]
        top6_plus1 = top6 + 1
        st.write(f"組{i+1}: {', '.join(map(str, top6_plus1))}")

# --- メイン処理 ---
def main():
    st.markdown("## CSVファイルアップロード")
    uploaded_file = st.file_uploader("過去のロト6抽選結果CSVファイルをアップロードしてください", type="csv")
    
    if uploaded_file is not None:
        # CSV内容のプレビューをエクスパンダーで表示
        df = load_data(uploaded_file)
        with st.expander("CSVデータプレビュー"):
            st.dataframe(df.head(10))
        
        # 分析開始ボタンと進捗表示
        if st.button("分析を開始する"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_progress = 0

            # 分析方法に応じた処理
            if analysis_method == "Gemini API":
                if gemini_api_key == "":
                    st.warning("Gemini API Keyをサイドバーに入力してください。")
                    return
                # ロト6の予測依頼プロンプトの作成
                total_draws = df.shape[0]
                sample_preview = df.head(3).to_string(index=False)
                lottery_prompt = (
                    f"以下は過去{total_draws}回分のロト6抽選結果のCSVデータの一部です。\n"
                    f"例:\n{sample_preview}\n\n"
                    "上記のデータに基づいて、次回のロト6抽選で出現する可能性が高いと思われる、"
                    "本数字6個の組み合わせを、必ず以下の形式で5組出力してください。\n"
                    "【出力例】\n"
                    "組1: 1, 2, 3, 4, 5, 6\n"
                    "組2: 7, 8, 9, 10, 11, 12\n"
                    "…\n"
                    "組5: 31, 32, 33, 34, 35, 36\n"
                    "※予測が不確実でも必ず5組出力してください。"
                )
                current_progress = 30
                progress_bar.progress(current_progress)
                status_text.text("Gemini APIへ予測依頼中...")
                predictions = get_gemini_predictions(gemini_api_key, lottery_prompt)
                current_progress = 80
                progress_bar.progress(current_progress)
                status_text.text("Gemini APIからの応答を取得中...")
                st.markdown("### Gemini API の予測結果")
                st.write(predictions)
            
            else:
                # 機械学習による解析の場合
                # ここでは、本数字部分（2列目～7列目）のみを特徴量として利用する例
                X = df.iloc[:, 1:7].values
                # ラベルは適当な値として、ここでは本数字と同じ内容を使う（実際のタスクでは変更してください）
                y = df.iloc[:, 1:7].values  
                X_scaled, y_encoded, n_classes = preprocess_data(X, y)
                # 1次元ラベルに変換（例としてOne-Hotエンコードのargmax）
                y_labels = y_encoded.argmax(axis=1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels
                )
                current_progress = 50
                progress_bar.progress(current_progress)
                status_text.text("データ分割中...")
                
                if analysis_method == "ニューラルネットワーク (単純)":
                    model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=64,
                        dropout=0.2,
                        learning_rate=1e-3,
                        n_classes=n_classes
                    )
                    epochs = 20
                    callback = ProgressBarCallback(progress_bar, epochs)
                    model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        verbose=0,
                        callbacks=[callback]
                    )
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.markdown("### ニューラルネットワークの評価結果")
                    st.write(f"テストデータでの損失: {loss:.4f}")
                    st.write(f"テストデータでの精度: {accuracy:.4f}")
                    pred_probs = model.predict(X_test[:5])
                    print_predicted_numbers_top6(pred_probs, n=5)
                
                elif analysis_method == "ランダムフォレスト":
                    rf_model = build_rf_model()
                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)
                    st.markdown("### ランダムフォレストの分類レポート")
                    st.text(classification_report(y_test, y_pred))
                    pred_probs = rf_model.predict_proba(X_test[:5])
                    print_predicted_numbers_top6(pred_probs, n=5)
                
                elif analysis_method == "Optuna + ニューラルネットワーク":
                    best_params = optimize_hyperparameters(X_train, y_train, n_classes)
                    st.markdown("### Optuna による最適パラメータ")
                    st.write(best_params)
                    best_model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=best_params['units'],
                        dropout=best_params['dropout'],
                        learning_rate=best_params['learning_rate'],
                        n_classes=n_classes
                    )
                    epochs = best_params['epochs']
                    batch_size = best_params['batch_size']
                    callback = ProgressBarCallback(progress_bar, epochs)
                    best_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[callback]
                    )
                    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
                    st.markdown("### Optuna + ニューラルネットワークの評価結果")
                    st.write(f"テストデータでの損失: {loss:.4f}")
                    st.write(f"テストデータでの精度: {accuracy:.4f}")
                    pred_probs = best_model.predict(X_test[:5])
                    print_predicted_numbers_top6(pred_probs, n=5)
            
            current_progress = 100
            progress_bar.progress(current_progress)
            status_text.text("処理完了！")
            
    else:
        st.info("まずCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
