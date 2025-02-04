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

# oneDNN無効化（必要に応じて）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

#############################
# データ読み込み＆前処理関数
#############################
def load_data(uploaded_file):
    """
    CSVファイルを読み込み、数値列のみ抜き出し、
    最後の列をラベルとして返す。
    """
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.error("CSVファイルには少なくとも2つの数値列が必要です。")
        st.stop()
    X = df[numeric_cols].iloc[:, :-1].values  # 特徴量
    y = df[numeric_cols].iloc[:, -1].values   # ラベル
    return np.array(X), np.array(y)

def preprocess_data(X, y):
    """
    データをスケーリングし、ラベルをOne-Hotエンコードした結果を返す。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # scikit-learnのバージョンにより、sparseまたはsparse_outputを指定します。
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    n_classes = y_encoded.shape[1]
    return X_scaled, y_encoded, n_classes

#############################
# モデル構築関数
#############################
def build_nn_model(input_dim, units, dropout, learning_rate, n_classes):
    """
    ニューラルネットワーク（全結合）モデルを構築
    """
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
    """
    ランダムフォレストモデルを構築
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

#############################
# ハイパーパラメータ最適化関数
#############################
def optimize_hyperparameters(X_train, y_train, n_classes):
    """
    Optunaでニューラルネットワークのハイパーパラメータを最適化
    """
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

#############################
# Gemini API呼び出し関数
#############################
def get_gemini_predictions(api_key, data):
    """
    Gemini APIにデータを送り、予測結果を受け取ります。
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{"text": "予測を生成するデータを送信"}]
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

#############################
# OpenAI o3-mini API呼び出し関数
#############################
def get_openai_o3mini_predictions(api_key, data):
    """
    OpenAI o3-mini APIにデータを送り、予測結果を受け取ります。

    Parameters:
        api_key (str): OpenAI APIの認証キー。
        data (str): 予測に使用するデータ（文字列化されたデータ）。

    Returns:
        str: 予測されたテキストの結果。
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"以下のデータに基づいて、予測結果を生成してください:\n{data}"
    payload = {
        "model": "o3-mini",  # 必要に応じて "o3-mini-high" を指定可能
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        prediction_text = result["choices"][0]["message"]["content"]
        return prediction_text
    except requests.exceptions.RequestException as e:
        st.error(f"OpenAI o3-mini APIエラー: {e}")
        return ""

#############################
# コールバッククラス
#############################
class ProgressBarCallback(tf.keras.callbacks.Callback):
    """
    ニューラルネットワーク学習の進捗をStreamlitのプログレスバーに反映するコールバック
    """
    def __init__(self, progress_bar, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress_rate = int(100 * self.current_epoch / self.total_epochs)
        self.progress_bar.progress(progress_rate)

#############################
# 予想数字の上位6クラスを5組出力
#############################
def print_predicted_numbers_top6(prob_array, n=5):
    """
    予測確率から、各サンプルの上位6クラスを5組表示する。
    """
    st.write("#### 予想される数字（各サンプルの上位6クラス）")
    for i in range(min(n, prob_array.shape[0])):
        sorted_indices = np.argsort(prob_array[i])[::-1]
        top6 = sorted_indices[:6]
        top6_plus1 = top6 + 1  # クラスが0始まりの場合、1を加算
        st.write(f"サンプル{i+1} → 予想数字: {list(top6_plus1)}")

#############################
# Streamlitアプリケーション
#############################
def main():
    st.set_page_config(page_title="ロト6データ分析アプリ", layout="wide")
    st.title("ロト6データ分析アプリ")

    # サイドバーにAPIキー入力欄を追加
    st.sidebar.header("APIキー設定")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

    # 以下はアプリ全体のフローイメージ（Mermaid記法）
    """
    ```mermaid
    flowchart TD
        A[ユーザー: CSVファイルアップロード]
        B[データ読み込み・前処理]
        C[分析方法選択]
        D{分析方法の分岐}
        E[ニューラルネットワーク／ランダムフォレスト学習]
        F[Optunaでハイパーパラメータ最適化]
        G[Gemini API呼び出し]
        H[OpenAI o3-mini API呼び出し]
        I[結果表示]
        
        A --> B
        B --> C
        C --> D
        D -- "従来の手法" --> E
        D -- "Optuna" --> F
        D -- "Gemini API" --> G
        D -- "OpenAI o3-mini" --> H
        E --> I
        F --> I
        G --> I
        H --> I
    ```
    """

    progress_bar = st.progress(0)
    status_text = st.empty()
    current_progress = 0

    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        try:
            current_progress = 10
            progress_bar.progress(current_progress)
            status_text.text("CSV読み込み中...")

            X, y = load_data(uploaded_file)
            st.success("データを正常に読み込みました！")

            current_progress = 30
            progress_bar.progress(current_progress)
            status_text.text("前処理を実行中...")

            analysis_method = st.radio(
                "分析方法を選択してください",
                ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク", "Gemini API", "OpenAI o3-mini")
            )

            if st.button("分析を開始する"):
                X_scaled, y_encoded, n_classes = preprocess_data(X, y)

                current_progress = 50
                progress_bar.progress(current_progress)
                status_text.text("データ分割中...")

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded.argmax(axis=1)
                )

                current_progress = 60
                progress_bar.progress(current_progress)
                status_text.text("モデルの学習またはAPI呼び出しを開始します...")

                if analysis_method == "Gemini API":
                    if gemini_api_key == "":
                        st.warning("Gemini API Keyをサイドバーに入力してください。")
                        return
                    predictions = get_gemini_predictions(gemini_api_key, X_test)
                    current_progress = 80
                    progress_bar.progress(current_progress)
                    status_text.text("Gemini APIからの予測結果を取得中...")
                    st.write("Gemini APIの予測結果:", predictions)

                elif analysis_method == "ニューラルネットワーク (単純)":
                    model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=64,
                        dropout=0.2,
                        learning_rate=1e-3,
                        n_classes=n_classes
                    )
                    epochs = 20
                    callback = ProgressBarCallback(progress_bar, epochs)
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        verbose=0,
                        callbacks=[callback]
                    )
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.write(f"テストデータでの損失: {loss:.4f}")
                    st.write(f"テストデータでの精度: {accuracy:.4f}")
                    pred_probs = model.predict(X_test[:5])
                    print_predicted_numbers_top6(pred_probs, n=5)

                elif analysis_method == "ランダムフォレスト":
                    rf_model = build_rf_model()
                    rf_model.fit(X_train, y_train.argmax(axis=1))
                    y_pred = rf_model.predict(X_test)
                    st.write("#### 分類レポート:")
                    st.text(classification_report(y_test.argmax(axis=1), y_pred))
                    if rf_model.n_classes_ > 1:
                        pred_probs = rf_model.predict_proba(X_test[:5])
                        if isinstance(pred_probs, list):
                            pred_probs = np.array(pred_probs).transpose(1, 0)
                        print_predicted_numbers_top6(pred_probs, n=5)

                elif analysis_method == "Optuna + ニューラルネットワーク":
                    best_params = optimize_hyperparameters(X_train, y_train, n_classes)
                    st.write("Optunaで探索された最適パラメータ:", best_params)
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
                    st.write(f"テストデータでの損失: {loss:.4f}")
                    st.write(f"テストデータでの精度: {accuracy:.4f}")
                    pred_probs = best_model.predict(X_test[:5])
                    print_predicted_numbers_top6(pred_probs, n=5)

                elif analysis_method == "OpenAI o3-mini":
                    if openai_api_key == "":
                        st.warning("OpenAI API Keyをサイドバーに入力してください。")
                        return
                    # 例としてX_testの先頭サンプルの数値データを文字列化して送信
                    sample_data = str(X_test[0].tolist())
                    prediction = get_openai_o3mini_predictions(openai_api_key, sample_data)
                    st.write("#### OpenAI o3-mini APIの予測結果:")
                    st.write(prediction)

                current_progress = 100
                progress_bar.progress(current_progress)
                status_text.text("完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
