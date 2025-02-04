def main():
    st.set_page_config(page_title="ロト6データ分析アプリ", layout="wide")
    st.title("ロト6データ分析アプリ")

    # サイドバーで API キー入力と高推論モードの切替
    st.sidebar.header("APIキー設定")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    use_high = st.sidebar.checkbox("高い推論 (o3-mini-high を利用)", value=False)

    # Mermaid 記法による全体フロー図（コメント内）
    """
    ```mermaid
    flowchart TD
        A[ユーザー: CSVファイルアップロード]
        B[データ読み込み・前処理]
        C[分析方法選択]
        D{分析方法の分岐}
        E[ニューラルネットワーク／ランダムフォレスト学習]
        F[Optunaでハイパーパラメータ最適化]
        G[Gemini API 呼び出し]
        H[OpenAI o3-mini API 呼び出し (SDK利用)]
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

    # CSVファイルアップロード
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
                    # 例として X_test の先頭サンプルの数値データを文字列に変換して送信
                    sample_data = str(X_test[0].tolist())
                    prediction = get_openai_o3mini_predictions_sdk(openai_api_key, sample_data, use_high=use_high)
                    st.write("#### OpenAI o3-mini APIの予測結果:")
                    st.write(prediction)

                current_progress = 100
                progress_bar.progress(current_progress)
                status_text.text("完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
