# LLM.py  (Streamlit main)

import streamlit as st
from PIL import Image
import pandas as pd
import io
import json
import google.generativeai as genai
import re
from io import BytesIO
import os
from pdf2image import convert_from_bytes

# -------------------- Utils --------------------

def setup_api_key():
    """Sidebar에서 Gemini API 키 입력/설정"""
    api_key = st.sidebar.text_input("Gemini API 키를 입력하세요.", type="password")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"유효하지 않은 API 키입니다. 다시 확인해 주세요: {e}")
            return False
    return False


@st.cache_data(show_spinner=False)
def parse_with_llm(image_data: bytes):
    """
    Gemini로 영수증 정보 추출. image_data는 JPEG 바이트.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction="""
            You are a world-class receipt and invoice parser.
            ONLY read the receipt itself; ignore background or other docs.

            Extract:
            1) date_time
            2) company_name
            3) business_number (10 digits, e.g., XXX-XX-XXXXX or digits only)
            4) address
            5) phone_number
            6) business_type

            Output strict JSON with these keys. Use null when missing.
            """
        )

        resp = model.generate_content(
            [{"mime_type": "image/jpeg", "data": image_data}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        json_string = resp.text or "{}"
        # 코드펜스 제거 대응
        json_string = re.sub(r'```json\s*|\s*```', '', json_string, flags=re.DOTALL)
        parsed = json.loads(json_string)
        # 키 누락 방지
        return {
            "date_time": parsed.get("date_time"),
            "company_name": parsed.get("company_name"),
            "business_number": parsed.get("business_number"),
            "address": parsed.get("address"),
            "phone_number": parsed.get("phone_number"),
            "business_type": parsed.get("business_type"),
        }
    except Exception as e:
        st.error(f"언어 모델 API 호출 중 오류 발생: {e}")
        return {
            "date_time": None,
            "company_name": None,
            "business_number": None,
            "address": None,
            "phone_number": None,
            "business_type": None,
        }


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300):
    """
    PDF → PIL.Image 리스트
    - 배포 환경(리눅스)에선 poppler-utils가 PATH에 설치되어 있어야 하며
      convert_from_bytes에 poppler_path를 절대 전달하지 않습니다.
    """
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def pil_image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -------------------- App --------------------

st.set_page_config(page_title="영수증 OCR", layout="centered")
st.title("📄 영수증 OCR 텍스트 추출기")
st.markdown("---")
st.write("JPG, PNG, PDF 파일을 여러 개 업로드하면 영수증 정보를 추출하고 CSV 파일로 다운로드할 수 있습니다.")

api_key_set = setup_api_key()

if api_key_set:
    uploaded_files = st.file_uploader(
        "파일을 선택하세요",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("읽어오기", use_container_width=True):
            all_rows = []
            progress = st.progress(0, text="파일 처리 중...")

            for idx, uf in enumerate(uploaded_files, start=1):
                ext = os.path.splitext(uf.name)[1].lower()

                with st.spinner(f"'{uf.name}' 처리 중..."):
                    # 업로드 바이트 확보 (포인터 문제 방지)
                    file_bytes = uf.getvalue()

                    try:
                        images = []
                        if ext == ".pdf":
                            images = pdf_to_images(file_bytes, dpi=300)
                        else:
                            # 단일 이미지도 리스트 처리로 통일
                            images = [Image.open(BytesIO(file_bytes))]

                        # 각 페이지/이미지별 LLM 파싱
                        for p, img in enumerate(images, start=1):
                            jpeg_bytes = pil_image_to_jpeg_bytes(img.convert("RGB"))
                            parsed = parse_with_llm(jpeg_bytes)

                            all_rows.append({
                                "File Name": uf.name if len(images) == 1 else f"{uf.name} - Page {p}",
                                "일시": parsed.get("date_time"),
                                "상호명": parsed.get("company_name"),
                                "사업자번호": parsed.get("business_number"),
                                "주소": parsed.get("address"),
                                "전화번호": parsed.get("phone_number"),
                                "업종": parsed.get("business_type"),
                            })

                    except Exception as e:
                        st.error(f"파일 처리 중 오류: {uf.name} — {e}")

                progress.progress(idx / len(uploaded_files), text=f"진행 중: {idx}/{len(uploaded_files)} 파일")

            progress.empty()
            st.markdown("---")

            if all_rows:
                df = pd.DataFrame(all_rows)
                st.subheader("✅ 추출 완료")
                st.dataframe(df, use_container_width=True)

                csv_io = io.StringIO()
                df.to_csv(csv_io, index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ CSV 파일로 다운로드",
                    data=csv_io.getvalue().encode("utf-8-sig"),
                    file_name="extracted_receipt_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.button("🔄 다시 시작하기", on_click=lambda: st.rerun(), use_container_width=True)
