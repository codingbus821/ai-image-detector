import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="AI 이미지 감지기", layout="centered")


@st.cache_resource
def load_model():
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # ❗ meta tensor 방지를 위해 low_cpu_mem_usage=False 사용
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=False  # 중요!
        )

        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None, None


def predict(image, processor, model):
    try:
        text = ["a photo", "an AI-generated image"]
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze()

        return probs
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return None


def main():
    st.title("🖼️ AI 이미지 감지기")
    st.write("업로드한 이미지가 실제 사진인지 AI가 생성한 이미지인지 판별합니다.")

    processor, model = load_model()
    if processor is None or model is None:
        return

    file = st.file_uploader("이미지를 업로드 해주세요.", type=["jpg", "jpeg", "png"])

    if file is None:
        st.warning("이미지를 업로드해주세요.")
    else:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="업로드한 이미지", use_container_width=True)

        with st.spinner("이미지 분석 중..."):
            probs = predict(image, processor, model)

        if probs is not None:
            st.subheader("결과")
            st.write(f"실사일 확률: **{probs[0].item():.2%}**")
            st.write(f"AI 생성 이미지일 확률: **{probs[1].item():.2%}**")

            if probs[1] > probs[0]:
                st.error("⚠️ 이 이미지는 AI가 생성했을 가능성이 높습니다.")
            else:
                st.success("✅ 이 이미지는 실사일 가능성이 높습니다.")


if __name__ == "__main__":
    main()
