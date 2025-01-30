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
# コールバッククラス
#############################
class ProgressBarCallback(tf.keras.callbacks.Callback):
    """
    ニューラルネットワーク学習の進捗を
    Streamlitのプログレスバーに反映させるコールバック
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
    予測確率 (prob_array) から、上位6クラスの数字を5組表示。
    prob_array.shape: (サンプル数, クラス数)
    n は表示するサンプル数（5組など）。
    """
    st.write("#### 予想される数字（各サンプルの上位6クラス）")
    for i in range(min(n, prob_array.shape[0])):
        # 確率が高い順にクラスをソート
        sorted_indices = np.argsort(prob_array[i])[::-1]
        top6 = sorted_indices[:6]  # 上位6つ
        # クラスが 0 始まりの場合、1 を足して「1～クラス数」の表記に
        top6_plus1 = top6 + 1
        st.write(f"サンプル{i+1} → 予想数字: {list(top6_plus1)}")

#############################
# Streamlitアプリケーション
#############################
def main():
    st.set_page_config(page_title="ロト6データ分析アプリ", layout="wide")
    st.title("ロト6データ分析アプリ")

    # プログレスバーとステータス用のUIパーツを準備
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_progress = 0

    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        try:
            # 進捗を10%に更新
            current_progress = 10
            progress_bar.progress(current_progress)
            status_text.text("CSV読み込み中...")

            X, y = load_data(uploaded_file)
            st.success("データを正常に読み込みました！")

            # 進捗を30%に更新
            current_progress = 30
            progress_bar.progress(current_progress)
            status_text.text("前処理を実行中...")

            # 分析方法の選択
            analysis_method = st.radio(
                "分析方法を選択してください",
                ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク", "Gemini API")
            )

            # 分析開始ボタン
            if st.button("分析を開始する"):
                X_scaled, y_encoded, n_classes = preprocess_data(X, y)

                # 進捗を50%に更新
                current_progress = 50
                progress_bar.progress(current_progress)
                status_text.text("データ分割中...")

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded.argmax(axis=1)
                )

                # 進捗を60%に更新
                current_progress = 60
                progress_bar.progress(current_progress)
                status_text.text("モデルの学習を開始します...")

                if analysis_method == "Gemini API":
                    gemini_api_key = st.secrets["GEMINI_API_KEY"]
                    predictions = get_gemini_predictions(gemini_api_key, X_test)

                    # 進捗を80%に更新
                    current_progress = 80
                    progress_bar.progress(current_progress)
                    status_text.text("Gemini APIからの予測結果を取得...")

                    st.write("Gemini APIの予測結果:", predictions)
                    # ここではAPIの戻り値形式次第で自由に表示

                elif analysis_method == "ニューラルネットワーク (単純)":
                    # ニューラルネットワークで単純に学習
                    model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=64,
                        dropout=0.2,
                        learning_rate=1e-3,
                        n_classes=n_classes
                    )

                    # コールバックを用意
                    epochs = 20
                    callback = ProgressBarCallback(progress_bar, epochs)

                    # 学習
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        verbose=0,
                        callbacks=[callback]
                    )

                    # 評価
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.write(f"テストデータでの損失: {loss:.4f}")
                    st.write(f"テストデータでの精度: {accuracy:.4f}")

                    # テストデータの先頭5サンプルから上位6クラスを予想数字として出力
                    pred_probs = model.predict(X_test[:5])  # shape: (5, n_classes)
                    print_predicted_numbers_top6(pred_probs, n=5)

                elif analysis_method == "ランダムフォレスト":
                    # ランダムフォレスト学習
                    rf_model = build_rf_model()
                    rf_model.fit(X_train, y_train.argmax(axis=1))  # yはOne-Hotなのでargmaxを取る

                    # 評価
                    y_pred = rf_model.predict(X_test)
                    st.write("#### 分類レポート:")
                    st.text(classification_report(y_test.argmax(axis=1), y_pred))

                    # 確率を取得して上位6クラスを5組出力 (クラス数があるときのみ)
                    if rf_model.n_classes_ > 1:
                        pred_probs = rf_model.predict_proba(X_test[:5])  # リスト(クラス数が多いと多次元になる)
                        # ランダムフォレストの predict_proba は「サンプル × (クラス数)」が返ってくる
                        # ただし 2クラス分類では shape=(5,2) のように1つだけになることも
                        # マルチクラスを想定している場合にのみ動作
                        if isinstance(pred_probs, list):
                            # 各クラスの確率が分割される場合、変形して結合する
                            pred_probs = np.array(pred_probs).transpose(1, 0)
                        print_predicted_numbers_top6(pred_probs, n=5)

                elif analysis_method == "Optuna + ニューラルネットワーク":
                    # Optunaでハイパーパラメータ探索
                    best_params = optimize_hyperparameters(X_train, y_train, n_classes)
                    st.write("Optunaで探索された最適パラメータ:", best_params)

                    # 最適パラメータでモデル再構築＆学習例 (任意)
                    best_model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=best_params['units'],
                        dropout=best_params['dropout'],
                        learning_rate=best_params['learning_rate'],
                        n_classes=n_classes
                    )
                    epochs = best_params['epochs']
                    batch_size = best_params['batch_size']

                    # 進捗バーコールバック（Optuna後に改めて学習するとき）
                    callback = ProgressBarCallback(progress_bar, epochs)

                    best_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[callback]
                    )

                    # 評価＆予想数字表示
                    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
                    st.write(f"テストデータでの損失: {loss:.4f}")
                    st.write(f"テストデータでの精度: {accuracy:.4f}")
                    pred_probs = best_model.predict(X_test[:5])
                    print_predicted_numbers_top6(pred_probs, n=5)

                # 最終的な進捗を100%に
                current_progress = 100
                progress_bar.progress(current_progress)
                status_text.text("完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
