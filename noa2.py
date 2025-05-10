import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class SkinDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load model
        self.model = load_model("/Users/macos/Documents/vnghia210.h5")
        self.class_names = ["Benh Zona", "Me Day", "Mun Coc", "Mun Trung Ca", "Nam Da", "Vay Nen", "Viem Da Co Dia"]
        self.INPUT_SIZE = (128, 128)
        self.MIN_SKIN_AREA = 2000
        self.CONFIDENCE_THRESHOLD = 85
        self.DETECTION_INTERVAL = 2.0  # Tăng thời gian giữa các lần phát hiện
        
        # Biến trạng thái
        self.current_results = []
        self.last_detection = 0
        self.camera_retry_count = 0
        self.max_retry_count = 3
        
        # Khởi tạo camera
        self.init_camera()
        
        # Khởi tạo UI
        self.init_ui()
    
    def init_camera(self):
        """Khởi tạo hoặc khởi tạo lại camera"""
        if hasattr(self, 'cap'):
            self.cap.release()
            time.sleep(0.5)  # Chờ giải phóng camera
            
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Thiết lập FPS cố định
        
    def init_ui(self):
        self.setWindowTitle("Chẩn đoán bệnh da liễu - Ổn định camera")
        self.setGeometry(100, 100, 1000, 600)
        
        # Widget chính
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout chính
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.central_widget.setLayout(self.main_layout)
        
        # Panel hiển thị camera
        self.camera_panel = QLabel()
        self.camera_panel.setAlignment(Qt.AlignCenter)
        self.camera_panel.setStyleSheet("background-color: black;")
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Panel thông tin
        self.info_panel = QWidget()
        self.info_layout = QVBoxLayout()
        self.info_layout.setContentsMargins(10, 10, 10, 10)
        self.info_panel.setLayout(self.info_layout)
        
        # Kết quả chẩn đoán
        self.disease_label = QLabel("Đang khởi động camera...")
        self.disease_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.disease_label.setAlignment(Qt.AlignCenter)
        
        self.status_label = QLabel("Trạng thái: Đang hoạt động")
        self.status_label.setStyleSheet("font-size: 14px;")
        
        self.info_layout.addWidget(self.disease_label)
        self.info_layout.addWidget(self.status_label)
        self.info_layout.addStretch()
        
        self.main_layout.addWidget(self.camera_panel, 3)
        self.main_layout.addWidget(self.info_panel, 1)
        
        # Timer để cập nhật hình ảnh
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS
    
    def update_frame(self):
        """Cập nhật frame từ camera với cơ chế phục hồi"""
        try:
            # Kiểm tra và khởi tạo lại camera nếu cần
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                if self.camera_retry_count < self.max_retry_count:
                    self.status_label.setText("Đang khởi động lại camera...")
                    self.init_camera()
                    self.camera_retry_count += 1
                    time.sleep(0.5)
                    return
                else:
                    self.status_label.setText("Lỗi: Không thể kết nối camera")
                    self.timer.stop()
                    return
            
            # Đọc frame từ camera
            ret, frame = self.cap.read()
            if not ret:
                self.camera_retry_count += 1
                if self.camera_retry_count >= self.max_retry_count:
                    self.status_label.setText("Lỗi: Không đọc được frame")
                    self.timer.stop()
                return
            
            # Reset retry count nếu đọc frame thành công
            self.camera_retry_count = 0
            
            # Giảm độ phân giải để tăng tốc độ xử lý
            frame = cv2.resize(frame, (640, 480))
            
            # Xử lý theo chu kỳ
            current_time = time.time()
            if current_time - self.last_detection > self.DETECTION_INTERVAL:
                self.process_detection(frame.copy())
                self.last_detection = current_time
            
            # Hiển thị trạng thái
            fps = 1/(time.time() - self.last_detection + 0.001)
            self.status_label.setText(f"Trạng thái: Hoạt động | FPS: {int(fps)}")
            
            # Hiển thị hình ảnh
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_panel.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            print(f"Lỗi trong update_frame: {str(e)}")
            self.status_label.setText(f"Lỗi: {str(e)}")
            if hasattr(self, 'cap'):
                self.cap.release()
            self.init_camera()
    
    def process_detection(self, frame):
        """Xử lý phát hiện bệnh với cơ chế bắt lỗi"""
        try:
            # Phát hiện vùng da
            skin_regions = self.detect_skin_regions(frame)
            
            # Dự đoán bệnh
            self.current_results = []
            for (x, y, w, h) in skin_regions:
                roi = frame[y:y+h, x:x+w]
                class_name, confidence = self.predict_skin_disease(roi)
                
                if class_name:
                    self.current_results.append({
                        "class": class_name,
                        "confidence": confidence,
                        "position": (x, y, w, h)
                    })
            
            # Cập nhật UI
            if self.current_results:
                best_result = max(self.current_results, key=lambda x: x["confidence"])
                self.disease_label.setText(f"{best_result['class']}")
                self.disease_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #27ae60;")
            else:
                self.disease_label.setText("Không phát hiện bệnh")
                self.disease_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c;")
                
        except Exception as e:
            print(f"Lỗi trong process_detection: {str(e)}")
            self.disease_label.setText("Lỗi khi xử lý")
    
    def detect_skin_regions(self, image):
        """Phát hiện vùng da với cơ chế đơn giản hơn"""
        try:
            # Chuyển đổi sang HSV và tạo mask
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Làm sạch mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Tìm contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            skin_regions = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.MIN_SKIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    skin_regions.append((x, y, w, h))
            
            return skin_regions
            
        except Exception as e:
            print(f"Lỗi trong detect_skin_regions: {str(e)}")
            return []
    
    def predict_skin_disease(self, roi):
        """Dự đoán bệnh với xử lý lỗi"""
        try:
            # Tiền xử lý nhanh
            resized = cv2.resize(roi, self.INPUT_SIZE)
            input_data = np.expand_dims(resized/255.0, axis=0)
            
            # Dự đoán
            predictions = self.model.predict(input_data, verbose=0)[0]
            confidence = np.max(predictions) * 100
            class_id = np.argmax(predictions)
            
            return (self.class_names[class_id], confidence) if confidence >= self.CONFIDENCE_THRESHOLD else (None, 0)
            
        except Exception as e:
            print(f"Lỗi trong predict_skin_disease: {str(e)}")
            return None, 0
    
    def closeEvent(self, event):
        """Đảm bảo giải phóng tài nguyên khi đóng ứng dụng"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = SkinDiseaseApp()
    window.show()
    app.exec_()