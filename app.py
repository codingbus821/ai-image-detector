
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import CLIPProcessor, CLIPModel

# 모델 및 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

st.title('AI 생성 이미지 판별기')
file = st.file_uploader('이미지를 업로드 해주세요.', type=['jpg', 'jpeg', 'png'])

if file is None:
    st.warning('이미지를 업로드해주세요.')
else:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    # CLIP에 넣기 위한 전처리
    inputs = processor(
        text=["a real photo", "an AI-generated image"],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0]

    # 결과 출력
    st.subheader("분석 결과")
    st.write(f"🟢 실사 이미지일 확률: **{probs[0]*100:.2f}%**")
    st.write(f"🔴 AI 생성 이미지일 확률: **{probs[1]*100:.2f}%**")

    if probs[1] > probs[0]:
        st.error("⚠️ 이 이미지는 AI가 생성했을 가능성이 높습니다.")
    else:
        st.success("✅ 이 이미지는 실사일 가능성이 높습니다.")
