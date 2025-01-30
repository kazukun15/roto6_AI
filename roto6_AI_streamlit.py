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
    # 正しいエンドポイントURLに更新
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{"text": "予測を生成するためのデータ"}]  # 必要に応じてデータ内容を変更
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
