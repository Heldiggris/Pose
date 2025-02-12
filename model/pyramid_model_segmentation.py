import tensorflow as tf
from tensorflow.keras.models import Model


class ChannelPadding(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ChannelPadding, self).__init__()
        self.channels = channels

    def build(self, input_shapes):
        self.pad_shape = tf.constant(
            [[0, 0], [0, 0], [0, 0], [0, self.channels - input_shapes[-1]]])

    def call(self, x):
        return tf.pad(x, self.pad_shape)


class Block(tf.keras.Model):
    def __init__(self, block_num=3, channel=48, channel_padding=1, name_prefix=""):
        super(Block, self).__init__()

        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(
                2, 2), padding='same', activation=None, name=name_prefix+"downsample_a_depthwise"),
            tf.keras.layers.Conv2D(
                filters=channel, kernel_size=1, activation=None, name=name_prefix+"downsample_a_conv1x1")
        ])
        if channel_padding:
            self.downsample_b = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                ChannelPadding(channels=channel)
            ])
        else:
            self.downsample_b = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv = list()
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, padding='same', activation=None, name=name_prefix+"conv_block_{}".format(i+1)),
                tf.keras.layers.Conv2D(
                    filters=channel, kernel_size=1, activation=None)
            ]))

    def call(self, x):
        x = tf.keras.activations.relu(
            self.downsample_a(x) + self.downsample_b(x))
        for i in range(len(self.conv)):
            x = tf.keras.activations.relu(x + self.conv[i](x))
        return x



class PyramidPoseModel():
    def __init__(self, num_keypoints: int):
        self.num_keypoints = num_keypoints

        # === Общие слои ===
        self.conv1 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=3, strides=(2, 2), padding='same', activation='relu'
        )
        self.conv2_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])
        self.conv2_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])

        # === Heatmap ===
        self.conv3 = Block(block_num=3, channel=48)
        self.conv4 = Block(block_num=4, channel=96)
        self.conv5 = Block(block_num=5, channel=192)
        self.conv6 = Block(block_num=6, channel=288)
        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])
        self.conv8a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])
        self.conv9a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])
        self.conv10a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv10b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu")
        ])
        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.Conv2D(
                filters=self.num_keypoints, kernel_size=3, padding="same", activation=None)  # Выход тепловой карты
        ])

        # === Регрессия ===
        self.conv12a = Block(block_num=4, channel=96, name_prefix="regression_conv12a_")
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv12b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=96, kernel_size=1, activation="relu", name="regression_conv12b_conv1x1")
        ], name="regression_conv12b")
        self.conv13a = Block(block_num=5, channel=192, name_prefix="regression_conv13a_")
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv13b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=192, kernel_size=1, activation="relu", name="regression_conv13b_conv1x1")
        ], name="regression_conv13b")
        self.conv14a = Block(block_num=6, channel=288, name_prefix="regression_conv14a_")
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv14b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=288, kernel_size=1, activation="relu", name="regression_conv14b_conv1x1")
        ], name="regression_conv14b")
        self.conv15 = tf.keras.models.Sequential([
            Block(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15a_"),
            Block(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15b_")
        ], name="regression_conv15")
        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=3 * self.num_keypoints, kernel_size=2, activation=None),
            tf.keras.layers.Reshape((3 * self.num_keypoints, 1), name="regression_final_dense")
        ], name="joints")

        # === Маска сегментации ===
        self.mask_branch = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding="same", activation="sigmoid")  # Выход маски
        ])

    def build_model(self, model_type):
        input_x = tf.keras.layers.Input(shape=(256, 256, 3))

        # === Общая часть ===
        x = self.conv1(input_x)
        x = x + self.conv2_1(x)
        x = tf.keras.activations.relu(x)
        x = x + self.conv2_2(x)
        y0 = tf.keras.activations.relu(x)

        # === Heatmap ===
        y1 = self.conv3(y0)
        y2 = self.conv4(y1)
        y3 = self.conv5(y2)
        y4 = self.conv6(y3)
        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        x = self.conv9a(x) + self.conv9b(y1)
        y = self.conv10a(x) + self.conv10b(y0)
        y = self.conv11(y)
        heatmap = tf.keras.layers.Activation("sigmoid", name="heatmap")(y)

        # === Регрессия ===
        if model_type == "ALL":
            x = tf.keras.backend.stop_gradient(x)
            y2 = tf.keras.backend.stop_gradient(y2)
            y3 = tf.keras.backend.stop_gradient(y3)
            y4 = tf.keras.backend.stop_gradient(y4)
        x = self.conv12a(x) + self.conv12b(y2)
        x = self.conv13a(x) + self.conv13b(y3)
        x = self.conv14a(x) + self.conv14b(y4)
        x = self.conv15(x)
        joints = self.conv16(x)

        # === Маска сегментации ===
        mask = self.mask_branch(y0)  # Используем выход после conv2_2

        # Возвращаем выходы
        if model_type == "ALL":
            return Model(inputs=input_x, outputs=[joints, heatmap, mask])
        elif model_type == "HEATMAP":
            return Model(inputs=input_x, outputs=heatmap)
        elif model_type == "REGRESSION":
            return Model(inputs=input_x, outputs=joints)
        elif model_type == "MASK":
            return Model(inputs=input_x, outputs=mask)
        else:
            raise ValueError("Wrong model type.")