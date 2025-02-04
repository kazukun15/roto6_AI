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
st.markdown("""
このアプリは、過去のロト6抽選結果のCSVデータに基づいて、  
次回のロト6抽選で出現する可能性が高いと予想される本数字6個の組み合わせを出力するデモです。  
※ Gemin APIによる予測と、機械学習（ニューラルネットワーク、ランダムフォレスト、Optuna最適化）の分析を選択できます。  
※ CSVの形式は「抽選回, 本数字1, 本数字2, ..., 本数字6, B数字, ｾｯﾄ」としてください。
""")

# --- oneDNNの最適化無効化 ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# --- サイドバー設定 ---
st.sidebar.header("【設定】")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Gemini APIキーを入力してください。")
st.sidebar.markdown("※ APIキーは本番環境では安全に管理してください。")
analysis_method = st.sidebar.radio(
    "分析方法を選択してください",
    ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク", "Gemini API")
)

# --- CSV読み込み関数 ---
def load_data(uploaded_file):
    """
    CSVファイルを読み込み、DataFrameとして返します。
    """
    df = pd.read_csv(uploaded_file, encoding="utf-8")
    return df

# --- 機械学習用関数群 ---
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 今回は y をそのまま使う簡易対応
    return X_scaled, y

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

# --- Gemini API 呼び出し関数 (修正版) ---
def get_gemini_predictions(api_key, prompt_text):
    """
    PaLM (Gemini) API に、指定のプロンプトを送り予測結果を取得します。
    現行の一般的な使用例として "generateText" を使用。
    """
    # 例: text-bison-001 や chat-bison-001 なども利用可能（要API確認）
    model_name = "text-bison-001"  # gemini-1.5-flash が使える場合はそちらを指定
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model_name}:generateText?key={api_key}"

    headers = {"Content-Type": "application/json"}

    # PaLM API では "prompt" を top-level に置き、"temperature" や "candidate_count" を指定する
    payload = {
        "prompt": {
            "text": prompt_text
        },
        "temperature": 0.2,
        "candidate_count": 1
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        # レスポンス本体は "candidates" キーが一般的
        candidates = response.json().get("candidates", [])
        # candidates が空でなければ、各候補の "output" に文章が格納されていることが多い
        outputs = [cand.get("output", "") for cand in candidates]
        return outputs
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

# --- 結果出力用関数 (機械学習用) ---
def print_predicted_numbers_top6(prob_array, n=5):
    st.write("#### 予想される数字（各組）")
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
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_progress = 0
            
            status_text.text("CSV読み込み中...")
            df = load_data(uploaded_file)
            st.success("CSVデータを正常に読み込みました！")
            
            with st.expander("CSVデータプレビュー"):
                st.dataframe(df.head(10))
            
            current_progress = 30
            progress_bar.progress(current_progress)
            status_text.text("分析準備中...")
            
            if st.button("分析を開始する"):
                if analysis_method == "Gemini API":
                    if gemini_api_key == "":
                        st.warning("Gemini API Keyをサイドバーに入力してください。")
                        return
                    # CSV全体を文字列として取得（長大すぎる場合は一部のみ使用）
                    csv_text = uploaded_file.getvalue().decode("utf-8")
                    max_chars = 4000  # 適宜変更
                    if len(csv_text) > max_chars:
                        csv_text = csv_text[:max_chars] + "\n...（以下省略）"
                    
                    # CSVの件数から情報を作成
                    total_draws = df.shape[0]
                    
                    lottery_prompt = (
                        f"以下は、過去{total_draws}回分のロト6抽選結果のCSVデータです。\n"
                        f"{csv_text}\n\n"
                        "上記のデータに基づいて、次回のロト6抽選で出現する可能性が高いと予想される、"
                        "本数字6個の組み合わせを、必ず以下の形式で5組出力してください。\n"
                        "【出力例】\n"
                        "組1: 1, 2, 3, 4, 5, 6\n"
                        "組2: 7, 8, 9, 10, 11, 12\n"
                        "…\n"
                        "組5: 31, 32, 33, 34, 35, 36\n"
                        "※予測が不確実でも必ず5組出力してください。"
                    )
                    
                    current_progress = 50
                    progress_bar.progress(current_progress)
                    status_text.text("Gemini APIへ予測依頼中...")
                    
                    # 修正版Gemini API呼び出し
                    predictions = get_gemini_predictions(gemini_api_key, lottery_prompt)
                    
                    current_progress = 80
                    progress_bar.progress(current_progress)
                    status_text.text("Gemini APIからの応答取得中...")

                    st.markdown("### Gemini API の予測結果")
                    if predictions:
                        # 1つの候補だけ返ってくる想定
                        for i, pred in enumerate(predictions, 1):
                            st.write(f"候補{i}:")
                            st.write(pred)
                    else:
                        st.warning("APIから有効な予測結果が得られませんでした。")
                
                else:
                    # 機械学習による分析の場合
                    # CSVの「本数字6個」部分（列2～7）を特徴量 X とし、
                    # ターゲットは便宜上そのうちの1列（例：列2）を使用
                    X = df.iloc[:, 1:7].values
                    y = df.iloc[:, 1].values
                    X_scaled, y_processed = preprocess_data(X, y)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42, stratify=y
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
                            n_classes=len(np.unique(y))
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
                    
                    elif analysis_method == "ランダムフォレスト":
                        rf_model = build_rf_model()
                        rf_model.fit(X_train, y_train)
                        y_pred = rf_model.predict(X_test)
                        st.markdown("### ランダムフォレストの分類レポート")
                        st.text(classification_report(y_test, y_pred))
                    
                    elif analysis_method == "Optuna + ニューラルネットワーク":
                        best_params = optimize_hyperparameters(X_train, y_train, len(np.unique(y)))
                        st.markdown("### Optuna による最適パラメータ")
                        st.write(best_params)
                        
                        best_model = build_nn_model(
                            input_dim=X_train.shape[1],
                            units=best_params['units'],
                            dropout=best_params['dropout'],
                            learning_rate=best_params['learning_rate'],
                            n_classes=len(np.unique(y))
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
                
                current_progress = 100
                progress_bar.progress(current_progress)
                status_text.text("処理完了！")
        
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.info("まずCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
