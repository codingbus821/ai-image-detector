import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def debug_device_info(tensors_dict):
    print("---- 디바이스 정보 ----")
    for k, v in tensors_dict.items():
        print(f"{k}: device={v.device}, shape={v.shape}, dtype={v.dtype}")
    print("---------------------")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 불러오기
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    # 모델 파라미터 디바이스 출력
    print("모델 파라미터 디바이스 예시:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
        break  # 한개만 확인

    # 이미지와 텍스트 준비 (디버깅용 더미 이미지 사용)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.new("RGB", (224, 224), color="red")
    inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt")

    # 입력 텐서 디바이스 확인
    debug_device_info(inputs)

    # 입력 텐서를 모델 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    debug_device_info(inputs)

    # 모델에 입력
    with torch.no_grad():
        outputs = model(**inputs)

    print("모델 출력:", outputs)

if __name__ == "__main__":
    main()
