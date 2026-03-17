import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 이미지 로드 (경로는 본인의 환경에 맞게 수정 가능)
    # 실험용사진1.jpeg 이미지를 사용합니다.
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', '실험용사진1.jpeg')
    
    # 한글 경로 인식을 위한 우회 읽기 방법
    try:
        with open(image_path, 'rb') as f:
            chunk = f.read()
        img_array = np.frombuffer(chunk, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"이미지 읽기 에러: {e}")
        return

    if img_bgr is None:
         print(f"이미지를 찾을 수 없습니다: {image_path}")
         return

    # OpenCV는 기본적으로 BGR(Blue, Green, Red) 순서로 이미지를 읽어옵니다.
    # matplotlib으로 올바르게 시각화하기 위해 RGB 순서로 변환합니다.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 첫 번째 방법: OpenCV 내장 함수를 이용한 Grayscale 변환
    # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)를 사용합니다.
    # 내부적으로 Gray = 0.299*R + 0.587*G + 0.114*B 공식을 사용하여
    # 사람의 눈이 녹색에 민감한 특성을 반영한 가중 평균을 구합니다.
    img_gray_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. 두 번째 방법: R, G, B 채널의 단순 평균을 이용한 Grayscale 변환
    # R, G, B 세 개의 채널을 직접 분리합니다.
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    
    # Gray = (R + G + B) / 3 공식으로 평균을 구한 후,
    # 다시 0~255 범위의 정수(uint8)로 변환합니다.
    img_gray_mean = ((R + G + B) / 3.0).astype(np.uint8)

    # 4. 결과 시각화 및 비교 (Presentation 화면)
    plt.figure(figsize=(15, 6))
    
    # 원본 이미지 출력
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('1. Original RGB Image', fontsize=14)
    plt.axis('off')
    
    # OpenCV 가중 평균 Grayscale 이미지 출력
    plt.subplot(1, 3, 2)
    plt.imshow(img_gray_cv2, cmap='gray')
    plt.title('2. OpenCV Grayscale\n(Weighted Average)', fontsize=14)
    plt.axis('off')
    
    # 단순 평균 Grayscale 이미지 출력
    plt.subplot(1, 3, 3)
    plt.imshow(img_gray_mean, cmap='gray')
    plt.title('3. Mean Grayscale\n((R+G+B)/3)', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # (추가 설명용) 두 Gray 이미지 간의 픽셀 값 차이 계산
    diff = cv2.absdiff(img_gray_cv2, img_gray_mean)
    mean_diff = np.mean(diff)
    print("=== 발표 설명용 요약 ===")
    print("Q. 두 변환 결과(OpenCV 방식 vs 단순 평균 방식)가 왜 다를까?")
    print("A. OpenCV 방식은 사람의 눈이 가장 민감하게 반응하는 초록빛(Green)에 더 높은 가중치(약 59%)를 주고, "
          "파란빛(Blue)에는 아주 낮은 가중치(약 11%)를 주는 '가중 평균(Weighted Average)'을 사용하기 때문입니다.")
    print("반면 단순 평균은 R, G, B 모두 33.3%의 동일한 가중치를 부여합니다.")
    print(f"-> 두 Gray 이미지 간의 평균 픽셀 밝기 차이: {mean_diff:.2f}")

if __name__ == '__main__':
    main()
