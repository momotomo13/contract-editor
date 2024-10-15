import streamlit as st
import PyPDF2
from openai import OpenAI
from io import BytesIO

# Streamlitアプリケーションの設定
st.set_page_config(page_title="契約書レビュー支援ツール", layout="wide")
st.title("契約書レビュー支援ツール")

# OpenAI クライアントの初期化
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# PDFファイルのアップロード
uploaded_file = st.file_uploader("契約書のPDFをアップロードしてください", type="pdf")

if uploaded_file is not None:
    with st.spinner("PDFを読み込んでいます..."):
        original_text = read_pdf(uploaded_file)

    st.success("PDFの読み込みが完了しました。分析を開始します...")

    # GPT-4による分析と修正
    with st.spinner("契約書を分析中..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": """あなたは経験豊富な企業法務の専門家です。契約書を分析し、法的リスクを最小限に抑え、両当事者の権利と義務を明確にする修正を提案してください。以下の点に注意して分析を行ってください：

1. 法的有効性：条項が法的に有効で強制力があることを確認
2. 明確性と具体性：曖昧な表現を避け、具体的で明確な言葉遣いを使用
3. リスク配分：リスクが適切に配分され、過度に一方の当事者に偏っていないことを確認
4. 整合性：契約書全体で用語の使用と条項間の整合性を確保
5. 法令遵守：適用される法律や規制に完全に準拠していることを確認
6. 紛争解決：紛争解決メカニズムが明確に定義されていることを確認
7. 秘密保持：必要に応じて、適切な秘密保持条項が含まれていることを確認
8. 契約終了：契約終了の条件と手続きが明確に規定されていることを確認

分析結果は以下の形式で提供してください：

1. 概要：契約書全体の評価と主要な問題点
2. 修正提案：
   a. 条項番号
   b. 現在の文言
   c. 修正提案
   d. 修正理由（法的根拠や潜在的リスクを含む）
   e. 重要度（高・中・低）

3. 追加推奨条項（必要な場合）
4. 総括：主要な修正点と全体的な改善の方向性

各修正提案は簡潔かつ具体的に記述し、法的な観点から重要な点を強調してください。"""},
                    {"role": "user", "content": f"以下の契約書を分析し、上記の指示に従って修正提案を提供してください：\n\n{original_text}"}
                ],
                max_tokens=4000
            )

            analysis_result = response.choices[0].message.content

            # 分析結果の表示
            st.subheader("契約書分析結果")
            st.write(analysis_result)

            # 分析結果をダウンロードするボタンを追加
            st.download_button(
                label="分析結果をダウンロード",
                data=analysis_result,
                file_name="contract_analysis_result.txt",
                mime="text/plain"
            )

            # 元の契約書の表示（オプション）
            with st.expander("元の契約書を表示"):
                st.subheader("元の契約書")
                st.code(original_text, language="plaintext")

        except Exception as e:
            st.error(f"分析中にエラーが発生しました: {str(e)}")

    st.success("分析が完了しました。")