import tensorflow as tf
import cv2
import numpy as np
import argparse

TEST_IMAGE = "data/test_image.jpg"
MODEL_PATH = "pre_train/model1.tflite"


class PoseDetector:
    def process_image(self, original_image, wait_time):
        h, w, _ = original_image.shape

        image = original_image.copy()  # Сохраняем копию исходного изображения
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
        image_resized = cv2.resize(image_color, (256, 256))  # Изменение размера до 256x256
        image_normalized = image_resized / 255.0  # Нормализация в [0, 1]
        image_input = np.expand_dims(image_normalized, axis=0).astype(np.float32)  # Добавление batch_size

        # Выполнение модели
        self.interpreter.set_tensor(self.input_details[0]['index'], image_input)
        self.interpreter.invoke()

        # Получение выходных данных
        keypoints_raw = self.interpreter.get_tensor(self.output_details[0]['index'])

        segmentation_mask = self.interpreter.get_tensor(self.output_details[2]['index'])

        # Постобработка ключевых точек
        keypoints = keypoints_raw.reshape(-1, 5)  # Преобразование в массив (N, 5)


        # Масштабирование точек обратно к размеру исходного изображения
        scaled_keypoints = []
        for x, y, z,_,_ in keypoints:
            scaled_x = int(w * x / 256) # Масштабирование по ширине
            scaled_y = int(h * y / 256) # Масштабирование по высоте
            scaled_keypoints.append((scaled_x, scaled_y, z))

        # Вывод исходного изображения с точками
        for x, y, _ in scaled_keypoints:
            cv2.circle(image, (x, y), radius=5, color=(0, 255, 0))  # Зелёная точка


        # Маска сегментации
        mask = segmentation_mask[0, :, :, 0]

        mask = np.where(mask < 0, 0, mask)        
        mask_resized = cv2.resize(mask, (w, h))

        cv2.imshow("Segmentation Mask", mask_resized)
        # Показать результат
        cv2.imshow("Image with Keypoints", image)
        cv2.waitKey(wait_time)
    
    def run(self, video):
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Получение информации о входах и выходах
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if isinstance(video, str):
            if video == "0":
                video = 0
            cap = cv2.VideoCapture(video)

            while cap.isOpened():
                ret, image = cap.read()
                
                if not ret:
                    break
                self.process_image(image, 1)
        else:
            # Загрузка изображения
            image = cv2.imread(TEST_IMAGE)
            self.process_image(image, 0)
            cv2.destroyAllWindows()




parser = argparse.ArgumentParser()
parser.add_argument(
    '-v',
    '--video',
    help='Работа с видео', default=-1)

args = parser.parse_args()

model = PoseDetector()
model.run(args.video)

