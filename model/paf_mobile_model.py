import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate

class PafMobileModel(Model):
    def __init__(self, num_keypoints: int):
        super(PafMobileModel, self).__init__()
        
        mob = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
        
        mob.trainable = False
        
        # Выбор выходов ранних слоёв
        self.backbone_output = mob.get_layer('block_5_add').output  # Разрешение (H/8, W/8)
        self.backbone = Model(inputs=mob.input, outputs=self.backbone_output)

        self.feature_adapter = tf.keras.Sequential([
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(128, kernel_size=3, padding='same', activation='relu')
        ])

        # Stages for PAF
        self.paf_stages = []
        for i in range(4):  # 4 этапа для PAF
            stage = self._make_stage(
                [128, 128, 128, 512, num_keypoints*2], kernel_sizes=[3, 3, 3, 1, 1], no_relu_last=True
            )
            self.paf_stages.append(stage)
        
        # Stages for Heatmaps
        self.heatmap_stages = []
        for i in range(2):  # 2 этапа для Heatmaps
            stage = self._make_stage(
                [128, 128, 128, 512, num_keypoints], kernel_sizes=[3, 3, 3, 1, 1], no_relu_last=True
            )
            self.heatmap_stages.append(stage)
    
    def _make_stage(self, filters, kernel_sizes, no_relu_last=False):
        layers = []
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            layers.append(Conv2D(f, kernel_size=k, padding='same'))
            if i < len(filters) - 1 or not no_relu_last:
                layers.append(tf.keras.layers.ReLU())
        return tf.keras.Sequential(layers)

    def call(self, x):
        # Backbone: извлечение признаков
        backbone_features = self.backbone(x)
        
        # Адаптация признаков
        out1 = self.feature_adapter(backbone_features)  # Улучшаем признаки
        
        # Вычисление PAF
        paf_out = out1
        for stage in self.paf_stages:
            paf_out = stage(paf_out)
        
        # Вычисление Heatmaps
        heatmap_out = paf_out  # Используем финальные PAF как вход для Heatmaps
        for stage in self.heatmap_stages:
            heatmap_out = stage(heatmap_out)
        
        return paf_out, heatmap_out
