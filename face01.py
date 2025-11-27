import cv2
import mediapipe as mp
import numpy as np
# 🌟 한글 출력을 위한 Pillow(PIL) 라이브러리 추가
from PIL import ImageFont, ImageDraw, Image

# 1. MediaPipe 설정
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 2. 웹캠 캡처 객체 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
    exit()

# 3. 폰트 설정 (Windows 기본 맑은 고딕 사용)
try:
    # 폰트 경로를 시스템에 맞게 지정하세요. (Windows 기준)
    font_path = "C:/Windows/Fonts/malgun.ttf" 
    # 폰트 크기를 조금 작게 조정하여 여러 줄이 보이도록 합니다.
    font = ImageFont.truetype(font_path, 20) 
except IOError:
    print("경고: 맑은 고딕 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    font = ImageFont.load_default()

# 4. 텍스트를 이미지에 삽입하는 함수 정의
def putText_korean(img, text, pos, font, color):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # PIL의 Draw 객체를 사용하여 한글 텍스트 삽입 (RGB 색상으로 변환)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0])) # BGR -> RGB 순서로 변환
    # PIL 이미지를 다시 OpenCV 이미지로 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# 5. 얼굴 메시 모델 초기화
with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    # 6. 실시간 비디오 스트림 처리 루프
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 이미지 처리 준비
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        ih, iw, _ = image.shape

        # 화면에 출력할 텍스트 리스트
        text_lines = [""] * 3 # 재물운, 애정운, 건강운 총 3줄
        line_colors = [(255, 255, 255)] * 3 # 기본 흰색
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                
                # 얼굴 특징점 (Landmarks) 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION, 
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                
                # --- A. 💰 재물운 (코, 재백궁) 분석 ---
                # 코의 폭을 나타내는 랜드마크 (358, 129)
                nose_wing_right = face_landmarks.landmark[358]
                nose_wing_left = face_landmarks.landmark[129]
                
                # 코의 폭 (정규화된 좌표 기준)
                nose_width = np.sqrt((nose_wing_right.x - nose_wing_left.x)**2 + (nose_wing_right.y - nose_wing_left.y)**2)
                
                if nose_width < 0.04: 
                    text_lines[0] = "💰 재물운: 코 폭이 좁아 금전 관리에 신중합니다."
                    line_colors[0] = (255, 255, 0) # 밝은 파랑
                elif nose_width > 0.05:
                    text_lines[0] = "💰 재물운: 코 폭이 넓어 재물복이 있고 활동적입니다."
                    line_colors[0] = (0, 255, 255) # 노란색
                else:
                    text_lines[0] = "💰 재물운: 코 모양이 균형 잡혀 재물운이 안정적입니다."
                    line_colors[0] = (0, 255, 0) # 녹색

                # --- B. ❤️ 애정운 (눈, 전택궁) 분석 ---
                # 눈 아래 (아래 눈꺼풀과 광대 사이, 전택궁 영역)
                eye_bottom_right = face_landmarks.landmark[145]
                eye_top_right = face_landmarks.landmark[159]
                
                # 눈 영역의 높이 (눈꺼풀과 눈 아래 거리)
                eye_height = np.sqrt((eye_bottom_right.x - eye_top_right.x)**2 + (eye_bottom_right.y - eye_top_right.y)**2)
                
                if eye_height < 0.015:
                    text_lines[1] = "❤️ 애정운: 눈 밑(전택궁)이 좁아 애정에 신중한 편입니다."
                    line_colors[1] = (255, 0, 255) # 마젠타
                elif eye_height > 0.03:
                    text_lines[1] = "❤️ 애정운: 눈 밑(전택궁)이 넓어 원만한 대인관계를 형성합니다."
                    line_colors[1] = (0, 0, 255) # 빨간색
                else:
                    text_lines[1] = "❤️ 애정운: 눈 주변이 밝아 좋은 인연을 맺을 운입니다."
                    line_colors[1] = (0, 255, 0) # 녹색
                
                # --- C. 🩺 건강운 (미간, 명궁) 분석 ---
                # 미간 좌우 (285, 55)
                forehead_right = face_landmarks.landmark[285]
                forehead_left = face_landmarks.landmark[55]
                
                # 미간의 너비
                forehead_width = np.sqrt((forehead_right.x - forehead_left.x)**2 + (forehead_right.y - forehead_left.y)**2)
                
                if forehead_width < 0.06:
                    text_lines[2] = "🩺 건강/기본운: 미간이 좁아 판단력이 빠르고 섬세합니다."
                    line_colors[2] = (255, 165, 0) # 주황색
                elif forehead_width > 0.08:
                    text_lines[2] = "🩺 건강/기본운: 미간이 넓어 성품이 여유롭고 건강합니다."
                    line_colors[2] = (0, 255, 255) # 노란색
                else:
                    text_lines[2] = "🩺 건강/기본운: 미간이 적당해 심신이 안정적입니다."
                    line_colors[2] = (0, 255, 0) # 녹색
                
                # 시각화를 위해 코 끝에 빨간색 원 표시 (재물운의 중심점)
                nose_tip = face_landmarks.landmark[1]
                nose_x = int(nose_tip.x * iw)
                nose_y = int(nose_tip.y * ih)
                cv2.circle(image, (nose_x, nose_y), 5, (0, 0, 255), -1)

        
        if not face_detected:
            # 얼굴 미감지 시 기본 메시지 설정
            text_lines = [""] * 3
            text_lines[0] = "얼굴을 카메라 중앙에 맞춰주세요."
            line_colors[0] = (255, 255, 255) # 흰색

        # 7. 관상 설명 텍스트를 화면 상단에 출력 (한글 처리)
        text_height = 25 # 한 줄당 차지하는 대략적인 높이
        
        # 텍스트 출력 공간 확보를 위한 검은색 배경
        max_text_height = (len(text_lines) * text_height) + 15 
        cv2.rectangle(image, (0, 0), (iw, max_text_height), (0, 0, 0), -1) 
        
        # 각 줄을 한글 처리 함수를 통해 출력
        for i, text in enumerate(text_lines):
            # i=0: 재물운 (Y=10), i=1: 애정운 (Y=35), i=2: 건강운 (Y=60)
            y_pos = 10 + (i * text_height)
            image = putText_korean(image, text, (10, y_pos), font, line_colors[i])

        # 8. 화면에 결과 프레임 표시
        cv2.imshow('Face Mesh Webcam (Press Q to quit)', image)
        
        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 9. 리소스 정리
cap.release()
cv2.destroyAllWindows()