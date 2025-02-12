import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate

class PafModel(Model):
    def __init__(self, num_keypoints: int):
        super(PafModel, self).__init__()
        
        self.no_relu_layers = [
            'conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
            'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
            'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
            'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L2'
        ]
        
        self.block0 = tf.keras.Sequential([
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(128, kernel_size=3, padding='same', activation='relu')
        ])
        
        # 1
        self.block1_1 = self._make_stage(
            [128, 128, 128, 512, num_keypoints*2], kernel_sizes=[3, 3, 3, 1, 1], no_relu_last=True
        )
        self.block1_2 = self._make_stage(
            [128, 128, 128, 512, num_keypoints], kernel_sizes=[3, 3, 3, 1, 1], no_relu_last=True
        )
        
        # 2-6
        self.stages = []
        for i in range(2, 3):
            stage_1 = self._make_stage_with_replacements(
                [128, 128, 128, 128, 128, 128, num_keypoints*2], kernel_sizes=[7, 7, 7, 7, 7, 1, 1], no_relu_last=True
            )
            stage_2 = self._make_stage_with_replacements(
                [128, 128, 128, 128, 128, 128, num_keypoints], kernel_sizes=[7, 7, 7, 7, 7, 1, 1], no_relu_last=True
            )
            self.stages.append((stage_1, stage_2))
    
    def _make_stage(self, filters, kernel_sizes, no_relu_last=False):
        layers = []
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            layers.append(Conv2D(f, kernel_size=k, padding='same'))
            if i < len(filters) - 1 or not no_relu_last:
                layers.append(tf.keras.layers.ReLU())
        return tf.keras.Sequential(layers)
    
    def _make_stage_with_replacements(self, filters, kernel_sizes, no_relu_last=False):
        layers = []
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            if k == 7:
                # Замена свёртки 7x7 на три свёртки 3x3
                layers.append(Conv2D(f, kernel_size=3, padding='same'))
                layers.append(tf.keras.layers.ReLU())
                layers.append(Conv2D(f, kernel_size=3, padding='same'))
                layers.append(tf.keras.layers.ReLU())
                layers.append(Conv2D(f, kernel_size=3, padding='same'))
            else:
                layers.append(Conv2D(f, kernel_size=k, padding='same'))
            if i < len(filters) - 1 or not no_relu_last:
                layers.append(tf.keras.layers.ReLU())
        return tf.keras.Sequential(layers)

    def call(self, x):
        out1 = self.block0(x)
        
        # 1
        out1_1 = self.block1_1(out1)  # PAF
        out1_2 = self.block1_2(out1)  # Heatmaps
        
        # 2-6
        out = tf.concat([out1_1, out1_2, out1], axis=-1)
        for stage_1, stage_2 in self.stages:
            out1_1 = stage_1(out)
            out1_2 = stage_2(out)
            out = tf.concat([out1_1, out1_2, out1], axis=-1)
        
        return out1_1, out1_2
    