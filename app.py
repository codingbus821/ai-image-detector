import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 모델 로딩
@st.cache_resource
def load_model():
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # 핵심: low_cpu_mem_usage=False를 명시하여 meta tensor 상태로 로드되지 않게 함
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", low_cpu_mem_usage=False)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None, None

def predict(image, processor, model):
    try:
        inputs = processor(
            text=["a photo", "an AI-generated image"],
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()
        return probs
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return None

def main():
    st.title("🖼️ AI 이미지 감별기 (디버깅용)")
    st.write("이미지를 업로드하면 AI가 생성한 이미지인지 판별합니다.")

    file = st.file_uploader("이미지를 업로드 해주세요.", type=["jpg", "jpeg", "png"])

    if file is None:
        st.warning("이미지를 업로드해주세요.")
        return

    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="업로드한 이미지", use_container_width=True)
    except Exception as e:
        st.error(f"이미지 처리 중 오류 발생: {e}")
        return

    processor, model = load_model()
    if processor is None or model is None:
        return

    with st.spinner("이미지 분석 중..."):
        probs = predict(image, processor, model)

    if probs:
        st.write(f"실사일 확률: {probs[0]*100:.2f}%")
        st.write(f"AI 생성 이미지일 확률: {probs[1]*100:.2f}%")

        if probs[1] > probs[0]:
            st.error("⚠️ 이 이미지는 AI가 생성했을 가능성이 높습니다.")
        else:
            st.success("✅ 이 이미지는 실사일 가능성이 높습니다.")

if __name__ == "__main__":
    main()
