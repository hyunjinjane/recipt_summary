import streamlit as st
from PIL import Image
import pandas as pd
import io
import json
import google.generativeai as genai
import re
from io import BytesIO
import os
from pdf2image import convert_from_bytes, exceptions

# Windows 사용자: 아래 변수에 Poppler의 bin 폴더 경로를 직접 입력하세요.
# 예시: r'C:\Users\username\poppler-0.68.0\bin'
poppler_path = r"C:\Users\PC\Desktop\캠프\Release-25.07.0-0\poppler-25.07.0\Library\bin"


def setup_api_key():
    """
    Sets up the API key from the user input.
    """
    api_key = st.sidebar.text_input("Gemini API 키를 입력하세요.", type="password")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"유효하지 않은 API 키입니다. 다시 확인해 주세요: {e}")
            return False
    st.info("왼쪽 사이드바에 Gemini API 키를 입력해주세요.")
    return False

@st.cache_data(show_spinner=False)
def parse_with_llm(image_data):
    """
    Uses the Gemini API to directly extract receipt information from an image.
    """
    try:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            system_instruction="""
            You are a world-class receipt and invoice parser.
            Your task is to extract specific information ONLY from the receipt part of the provided image.
            **IMPORTANT:** The image might contain irrelevant information from the background or other documents. You must ignore all content that is not part of the main receipt. Focus exclusively on the structured receipt data.

            Extract the following fields accurately:
            1.  **Date and Time (일시):** The date and time of the transaction. Look for patterns like YYYY.MM.DD, YYYY-MM-DD, or MM/DD/YYYY and time formats like HH:MM. If both are found, combine them.
            2.  **Company Name (상호명):** The name of the business or store. It is crucial to get this field correct. Be very careful with this and try to infer the correct name from partial or misspelled words. Also, consider the name that appears most prominently at the top of the document.
            3.  **Business Number (사업자번호):** The 10-digit business registration number, often in the format XXX-XX-XXXXX.
            4.  **Address (주소):** The street address of the business.
            5.  **Phone Number (전화번호):** The phone number of the business.
            6.  **Business Type (업종):** Infer the type of business from the company name, items purchased, or other context in the text. (e.g., Restaurant, Cafe, Retail, etc.).

            The final output must be a JSON object with the following keys. If a piece of data is not found, use `null`.
            - `date_time` (string)
            - `company_name` (string)
            - `business_number` (string)
            - `address` (string)
            - `phone_number` (string)
            - `business_type` (string)
            """
        )

        # Pass the image data to the API
        response = model.generate_content(
            [{"mime_type": "image/jpeg", "data": image_data}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        json_string = response.text
        json_string = re.sub(r'```json\s*|\s*```', '', json_string, flags=re.DOTALL)
        parsed_data = json.loads(json_string)
        return parsed_data
    except Exception as e:
        st.error(f"언어 모델 API 호출 중 오류 발생: {e}")
        return {
            "date_time": None,
            "company_name": None,
            "business_number": None,
            "address": None,
            "phone_number": None,
            "business_type": None
        }

# Streamlit page configuration
st.set_page_config(page_title="영수증 OCR", layout="centered")

# --- UI Layout Start ---
st.title("📄 영수증 OCR 텍스트 추출기")
st.markdown("---")
st.write("JPG, PNG, PDF 파일을 여러 개 업로드하면 영수증 정보를 추출하고 CSV 파일로 다운로드할 수 있습니다.")

api_key_set = setup_api_key()

if api_key_set:
    uploaded_files = st.file_uploader("파일을 선택하세요", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("읽어오기", use_container_width=True):
            all_extracted_data = []
            progress_bar = st.progress(0, text="파일 처리 중...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                with st.spinner(f"'{uploaded_file.name}' 파일 처리 중..."):
                    file_bytes = uploaded_file.getvalue()
                    
                    # PDF는 이미지로 변환해서 처리해야 합니다.
                    if file_extension == ".pdf":
                        try:
                            images = convert_from_bytes(file_bytes, poppler_path=poppler_path)
                            
                            for image_obj in images:
                                image_bytes = BytesIO()
                                image_obj.save(image_bytes, format='JPEG')
                                parsed_info = parse_with_llm(image_bytes.getvalue())
                                all_extracted_data.append({
                                    "File Name": f"{uploaded_file.name} - Page {images.index(image_obj)+1}",
                                    "일시": parsed_info.get("date_time"),
                                    "상호명": parsed_info.get("company_name"),
                                    "사업자번호": parsed_info.get("business_number"),
                                    "주소": parsed_info.get("address"),
                                    "전화번호": parsed_info.get("phone_number"),
                                    "업종": parsed_info.get("business_type")
                                })
                        except exceptions.PopplerNotInstalledError:
                            st.error("Poppler가 설치되지 않았습니다. Windows 사용자는 Poppler를 설치하고 poppler_path 변수를 올바르게 설정해야 합니다.")
                            st.stop()
                        except Exception as e:
                            st.error(f"PDF 파일 처리 중 오류 발생: {e}")
                            continue
                    else:
                        parsed_info = parse_with_llm(file_bytes)
                        all_extracted_data.append({
                            "File Name": uploaded_file.name,
                            "일시": parsed_info.get("date_time"),
                            "상호명": parsed_info.get("company_name"),
                            "사업자번호": parsed_info.get("business_number"),
                            "주소": parsed_info.get("address"),
                            "전화번호": parsed_info.get("phone_number"),
                            "업종": parsed_info.get("business_type")
                        })

            progress_bar.progress((i + 1) / len(uploaded_files), text=f"진행 중: {i+1}/{len(uploaded_files)} 파일")
        
        progress_bar.empty()
        
        st.markdown("---")
        
        if all_extracted_data:
            df = pd.DataFrame(all_extracted_data)
            
            st.subheader("✅ 추출 완료")
            st.dataframe(df, use_container_width=True)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_data = csv_buffer.getvalue().encode('utf-8-sig')
            
            st.download_button(
                label="⬇️ CSV 파일로 다운로드",
                data=csv_data,
                file_name='extracted_receipt_data.csv',
                mime='text/csv',
                use_container_width=True
            )
        
st.button("🔄 다시 시작하기", on_click=lambda: st.rerun(), use_container_width=True)
