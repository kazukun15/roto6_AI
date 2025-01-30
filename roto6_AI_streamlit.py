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
from sklearn.metrics import classification_report, f1_score, roc_auc_score
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

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # クラス数
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
        Dense(n_classes, activation='softmax')  # クラス数を動的に設定
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

        # 学習
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # 訓練データでの精度（暫定指標）
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

    Parameters:
        api_key (str): Gemini APIの認証キー。
        data (list or np.ndarray): 予測に使用するデータ。

    Returns:
        list: 予測された番号のリスト。
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
# Streamlitアプリケーション
#############################
def main():
    st.set_page_config(page_title="ロト6データ分析アプリ", layout="wide")
    st.title("ロト6データ分析アプリ")

    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        try:
            X, y = load_data(uploaded_file)
            st.success("データを正常に読み込みました！")

            # 分析方法の選択
            analysis_method = st.radio(
                "分析方法を選択してください",
                ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク", "Gemini API")
            )

            if st.button("分析を開始する"):
                X_scaled, y_encoded, n_classes = preprocess_data(X, y)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded.argmax(axis=1)
                )

                if analysis_method == "Gemini API":
                    gemini_api_key = st.secrets["GEMINI_API_KEY"]
                    predictions = get_gemini_predictions(gemini_api_key, X_test)
                    st.write("Gemini APIの予測結果:", predictions)

                else:
                    st.write("他の分析方法が選択されました")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
