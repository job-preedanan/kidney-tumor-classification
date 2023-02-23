import tensorflow as tf


def make_mulmoNet_vgg19(height, width, channels, output_channels):
    # load pretrained model
    base_model1 = tf.keras.applications.vgg19.VGG19(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg19.VGG19(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False
    x1 = base_model1.output
    x2 = base_model2.output

    x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(output_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=x)

    return model


def make_mulmoNet_vgg16(encoder_num=3, height=224, width=224, channels=3, output_channels=1):
    # load pretrained model
    # encoder 1
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model1.trainable = False
    latent_1 = base_model1.get_layer('block5_conv3_m1').output

    # encoder 2
    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model2.trainable = False
    latent_2 = base_model2.get_layer('block5_conv3_m2').output

    # encoder 3
    if encoder_num == 3:
        base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

        for layer in base_model3.layers:
            layer._name = layer.name + str('_m3')

        base_model3.trainable = False
        latent_3 = base_model3.get_layer('block5_conv3_m3').output

    # encoder 5
    if encoder_num == 5:
        base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

        for layer in base_model3.layers:
            layer._name = layer.name + str('_m3')

        base_model3.trainable = False
        latent_3 = base_model3.get_layer('block5_conv3_m3').output

        base_model4 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

        for layer in base_model4.layers:
            layer._name = layer.name + str('_m4')

        base_model4.trainable = False
        latent_4 = base_model4.get_layer('block5_conv3_m4').output

        base_model5 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

        for layer in base_model5.layers:
            layer._name = layer.name + str('_m5')

        base_model5.trainable = False
        latent_5 = base_model5.get_layer('block5_conv3_m5').output

    if encoder_num == 2:
        concate_latent = tf.keras.layers.Concatenate(axis=-1)([latent_1, latent_2])
    elif encoder_num == 3:
        concate_latent = tf.keras.layers.Concatenate(axis=-1)([latent_1, latent_2, latent_3])
    elif encoder_num == 5:
        concate_latent = tf.keras.layers.Concatenate(axis=-1)([latent_1, latent_2, latent_3, latent_4, latent_5])

    dense = tf.keras.layers.GlobalAveragePooling2D()(concate_latent)
    dense = tf.keras.layers.Dense(4096, activation='relu')(dense)
    dense = tf.keras.layers.Dense(4096, activation='relu')(dense)
    # output = tf.keras.layers.Dense(output_channels, activation='sigmoid')(dense)
    output = tf.keras.layers.Dense(5, activation='softmax')(dense)

    model = tf.keras.Model(inputs=[base_model1.input,
                                   base_model2.input,
                                   base_model3.input], outputs=output)

    return model


def make_mulmoUNet_vgg16(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model 1
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    # load pretrained model 2
    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    # load pretrained model 3
    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2, lat_output3])

    # -------------------------------------------DECODER PART ------------------------------------------------------

    enc0 = base_model1.get_layer('block1_conv2_m1').output
    enc1 = base_model1.get_layer('block2_conv2_m1').output
    enc2 = base_model1.get_layer('block3_conv2_m1').output
    enc3 = base_model1.get_layer('block4_conv3_m1').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                           kernel_size=2,
                                           strides=2,
                                           kernel_initializer='he_uniform')(vgg_output_concate)
    dec3 = tf.keras.layers.BatchNormalization()(dec3)
    dec3 = tf.keras.layers.Concatenate(axis=-1)([dec3, enc3])

    dec2 = decoding_block(first_layer_filter_count * 4, dec3)  # (N/4 x N/4 x 4CH)
    dec2 = tf.keras.layers.Concatenate(axis=-1)([dec2, enc2])

    dec1 = decoding_block(first_layer_filter_count * 2, dec2)  # (N/2 x N/2 x 2CH)
    dec1 = tf.keras.layers.Concatenate(axis=-1)([dec1, enc1])

    dec0 = decoding_block(first_layer_filter_count, dec1)  # (N x N x CH)
    dec0 = tf.keras.layers.Concatenate(axis=-1)([dec0, enc0])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    last = convolution_block(first_layer_filter_count, dec0)
    last = convolution_block(first_layer_filter_count, last)
    last = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(last)
    last = tf.keras.layers.Activation(activation='sigmoid')(last)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input], outputs=last)

    return model


def make_mulmoXNet_vgg16(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, enc3_m1])

    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_m1)  # (N/4 x N/4 x 4CH)
    dec2_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, enc2_m1])

    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_m1)  # (N/2 x N/2 x 2CH)
    dec1_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, enc1_m1])

    dec0_m1 = decoding_block(first_layer_filter_count, dec1_m1)  # (N x N x CH)
    dec0_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, enc0_m1])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # -------------------------------------------DECODER 2 ------------------------------------------------------

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, enc3_m2])

    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_m2)  # (N/4 x N/4 x 4CH)
    dec2_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, enc2_m2])

    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_m2)  # (N/2 x N/2 x 2CH)
    dec1_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, enc1_m2])

    dec0_m2 = decoding_block(first_layer_filter_count, dec1_m2)  # (N x N x CH)
    dec0_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, enc0_m2])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=[output1, output2])

    return model


def make_new_mulmoXNet_vgg16_2(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_concate = tf.keras.layers.Concatenate(axis=-1)([enc0_m1, enc0_m2])
    enc1_concate = tf.keras.layers.Concatenate(axis=-1)([enc1_m1, enc1_m2])
    enc2_concate = tf.keras.layers.Concatenate(axis=-1)([enc2_m1, enc2_m2])
    enc3_concate = tf.keras.layers.Concatenate(axis=-1)([enc3_m1, enc3_m2])

    first_layer_filter_count = 64

    # vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, enc3_concate])

    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_m1)  # (N/4 x N/4 x 4CH)
    dec2_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, enc2_concate])

    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_m1)  # (N/2 x N/2 x 2CH)
    dec1_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, enc1_concate])

    dec0_m1 = decoding_block(first_layer_filter_count, dec1_m1)  # (N x N x CH)
    dec0_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, enc0_concate])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # -------------------------------------------DECODER 2 ------------------------------------------------------

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, enc3_concate])

    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_m2)  # (N/4 x N/4 x 4CH)
    dec2_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, enc2_concate])

    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_m2)  # (N/2 x N/2 x 2CH)
    dec1_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, enc1_concate])

    dec0_m2 = decoding_block(first_layer_filter_count, dec1_m2)  # (N x N x CH)
    dec0_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, enc0_concate])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=[output1, output2])

    return model


def make_new_mulmoXNet_vgg16_3(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    # enc0_concate = tf.keras.layers.Concatenate(axis=-1)([enc0_m1, enc0_m2])
    # enc1_concate = tf.keras.layers.Concatenate(axis=-1)([enc1_m1, enc1_m2])
    # enc2_concate = tf.keras.layers.Concatenate(axis=-1)([enc2_m1, enc2_m2])
    # enc3_concate = tf.keras.layers.Concatenate(axis=-1)([enc3_m1, enc3_m2])

    first_layer_filter_count = 64

    # vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)


    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, enc3_m1])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, enc3_m2])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, enc2_m1])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, enc2_m2])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, enc1_m1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, enc1_m2])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, enc0_m1])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, enc0_m2])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=[output1, output2])

    return model


def make_new_mulmoXNet_vgg16_4(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_concate = tf.keras.layers.Concatenate(axis=-1)([enc0_m1, enc0_m2])
    enc1_concate = tf.keras.layers.Concatenate(axis=-1)([enc1_m1, enc1_m2])
    enc2_concate = tf.keras.layers.Concatenate(axis=-1)([enc2_m1, enc2_m2])
    enc3_concate = tf.keras.layers.Concatenate(axis=-1)([enc3_m1, enc3_m2])

    first_layer_filter_count = 64

    # vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, enc3_concate])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, enc3_concate])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, enc2_concate])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, enc2_concate])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, enc1_concate])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, enc1_concate])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec2_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec2_concat_m2)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, enc0_concate])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, enc0_concate])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=[output1, output2])

    return model


def make_new_mulmoXNet_vgg16_5(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_concate = tf.keras.layers.Concatenate(axis=-1)([enc0_m1, enc0_m2])
    enc1_concate = tf.keras.layers.Concatenate(axis=-1)([enc1_m1, enc1_m2])
    enc2_concate = tf.keras.layers.Concatenate(axis=-1)([enc2_m1, enc2_m2])
    enc3_concate = tf.keras.layers.Concatenate(axis=-1)([enc3_m1, enc3_m2])

    first_layer_filter_count = 64

    vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, enc3_concate])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, enc3_concate])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, enc2_concate])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, enc2_concate])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, enc1_concate])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, enc1_concate])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec2_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec2_concat_m2)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, enc0_concate])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, enc0_concate])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=[output1, output2])

    return model


def make_mulmo_3encUNet_vgg16(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model 1
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    # load pretrained model 2
    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    # load pretrained model 3
    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1,
                                                               lat_output2,
                                                               lat_output3])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, enc3_m1])

    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_m1)  # (N/4 x N/4 x 4CH)
    dec2_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, enc2_m1])

    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_m1)  # (N/2 x N/2 x 2CH)
    dec1_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, enc1_m1])

    dec0_m1 = decoding_block(first_layer_filter_count, dec1_m1)  # (N x N x CH)
    dec0_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, enc0_m1])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # -------------------------------------------DECODER 2 ------------------------------------------------------

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    # (N/8 x N/8 x 8CH)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, enc3_m2])

    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_m2)  # (N/4 x N/4 x 4CH)
    dec2_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, enc2_m2])

    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_m2)  # (N/2 x N/2 x 2CH)
    dec1_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, enc1_m2])

    dec0_m2 = decoding_block(first_layer_filter_count, dec1_m2)  # (N x N x CH)
    dec0_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, enc0_m2])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # -------------------------------------------DECODER 3 ------------------------------------------------------

    enc0_m3 = base_model3.get_layer('block1_conv2_m3').output
    enc1_m3 = base_model3.get_layer('block2_conv2_m3').output
    enc2_m3 = base_model3.get_layer('block3_conv2_m3').output
    enc3_m3 = base_model3.get_layer('block4_conv3_m3').output

    # (N/8 x N/8 x 8CH)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)
    dec3_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m3, enc3_m3])

    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_m3)  # (N/4 x N/4 x 4CH)
    dec2_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m3, enc2_m3])

    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_m3)  # (N/2 x N/2 x 2CH)
    dec1_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m3, enc1_m3])

    dec0_m3 = decoding_block(first_layer_filter_count, dec1_m3)  # (N x N x CH)
    dec0_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m3, enc0_m3])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                           outputs=[output1, output2, output3])

    return model


def make_mulmo_3encUNet_vgg16_nested_decoder(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2, lat_output3])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_m3 = base_model3.get_layer('block1_conv2_m3').output
    enc1_m3 = base_model3.get_layer('block2_conv2_m3').output
    enc2_m3 = base_model3.get_layer('block3_conv2_m3').output
    enc3_m3 = base_model3.get_layer('block4_conv3_m3').output

    # enc0_concate = tf.keras.layers.Concatenate(axis=-1)([enc0_m1, enc0_m2])
    # enc1_concate = tf.keras.layers.Concatenate(axis=-1)([enc1_m1, enc1_m2])
    # enc2_concate = tf.keras.layers.Concatenate(axis=-1)([enc2_m1, enc2_m2])
    # enc3_concate = tf.keras.layers.Concatenate(axis=-1)([enc3_m1, enc3_m2])

    first_layer_filter_count = 64

    # vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m1])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m2])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m1])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m3])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m2])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m3])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m1])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m2])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m3])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                           outputs=[output1, output2, output3])

    return model


def make_mulmo_3encUNet_vgg16_nested_decoder_add(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2, lat_output3])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_m3 = base_model3.get_layer('block1_conv2_m3').output
    enc1_m3 = base_model3.get_layer('block2_conv2_m3').output
    enc2_m3 = base_model3.get_layer('block3_conv2_m3').output
    enc3_m3 = base_model3.get_layer('block4_conv3_m3').output

    # enc0_concate = tf.keras.layers.Concatenate(axis=-1)([enc0_m1, enc0_m2])
    # enc1_concate = tf.keras.layers.Concatenate(axis=-1)([enc1_m1, enc1_m2])
    # enc2_concate = tf.keras.layers.Concatenate(axis=-1)([enc2_m1, enc2_m2])
    # enc3_concate = tf.keras.layers.Concatenate(axis=-1)([enc3_m1, enc3_m2])

    first_layer_filter_count = 64

    # vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)
    dec3_add = tf.keras.layers.Add()([dec3_m1, dec3_m2, dec3_m3])
    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_add, enc3_m1])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_add, enc3_m2])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_add, enc3_m3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_add = tf.keras.layers.Add()([dec2_m1, dec2_m2, dec2_m3])
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_add, enc2_m1])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_add, enc2_m2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_add, enc2_m3])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_add = tf.keras.layers.Add()([dec1_m1, dec1_m2, dec1_m3])
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_add, enc1_m1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_add, enc1_m2])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_add, enc1_m3])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_add = tf.keras.layers.Add()([dec0_m1, dec0_m2, dec0_m3])
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_add, enc0_m1])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_add, enc0_m2])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_add, enc0_m3])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                           outputs=[output1, output2, output3])

    return model

def make_mulmo_3encUNet_vgg16_nested_decoder_att(height, width, channels, output_channels):
    def attention_block(sequence, gating, filter_count):

        # reshape x and gating layers
        x = tf.keras.layers.Conv2D(filter_count, kernel_size=1, strides=2, padding='same')(sequence)
        gating = tf.keras.layers.Conv2D(filter_count, kernel_size=1, strides=1, padding='same')(gating)

        # add x and g + ReLU
        concate_xg = tf.keras.layers.add([x, gating])
        relu_xg = tf.keras.layers.ReLU()(concate_xg)

        # conv(nb_filter=1) + sigmoid
        conv1_xg = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(relu_xg)
        sigmoid_xg = tf.keras.layers.Activation(activation='sigmoid')(conv1_xg)

        # upsample
        upsample_sigmoid_xg = tf.keras.layers.UpSampling2D(size=(2, 2))(sigmoid_xg)

        # multiply with x and conv(nb_filters=x_filters)
        y = tf.keras.layers.multiply([upsample_sigmoid_xg, sequence])
        conv1_xg = tf.keras.layers.Conv2D(filter_count, kernel_size=1, strides=1, padding='same')(y)
        new_sequence = tf.keras.layers.BatchNormalization()(conv1_xg)

        return new_sequence

    def decoding_block(filter_count, sequence, residual=False):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        if residual:
            shortcut = tf.keras.layers.Conv2D(filter_count * 2,
                                              kernel_size=1,
                                              strides=1,
                                              padding='same')(sequence)
            new_sequence = tf.keras.layers.add([shortcut, new_sequence])

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2, lat_output3])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_m3 = base_model3.get_layer('block1_conv2_m3').output
    enc1_m3 = base_model3.get_layer('block2_conv2_m3').output
    enc2_m3 = base_model3.get_layer('block3_conv2_m3').output
    enc3_m3 = base_model3.get_layer('block4_conv3_m3').output

    first_layer_filter_count = 64

    # vgg_output_concate = convolution_block(first_layer_filter_count * 8, vgg_output_concate)

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)

    enc3_m1 = attention_block(enc3_m1, vgg_output_concate, first_layer_filter_count * 8)
    enc3_m2 = attention_block(enc3_m2, vgg_output_concate, first_layer_filter_count * 8)
    enc3_m3 = attention_block(enc3_m3, vgg_output_concate, first_layer_filter_count * 8)
    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m1])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m2])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1, residual=True)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2, residual=True)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3, residual=True)
    enc2_m1 = attention_block(enc2_m1, dec3_concat_m1, first_layer_filter_count * 4)
    enc2_m2 = attention_block(enc2_m2, dec3_concat_m2, first_layer_filter_count * 4)
    enc2_m3 = attention_block(enc2_m3, dec3_concat_m3, first_layer_filter_count * 4)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m1])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m3])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1, residual=True)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2, residual=True)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3, residual=True)
    enc1_m1 = attention_block(enc1_m1, dec2_concat_m1, first_layer_filter_count * 2)
    enc1_m2 = attention_block(enc1_m2, dec2_concat_m2, first_layer_filter_count * 2)
    enc1_m3 = attention_block(enc1_m3, dec2_concat_m3, first_layer_filter_count * 2)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m2])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3,enc1_m3])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1, residual=True)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2, residual=True)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3, residual=True)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m1])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m2])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m3])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1_0 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1_1 = convolution_block(first_layer_filter_count, output1_0)
    shortcut1 = tf.keras.layers.Conv2D(first_layer_filter_count, kernel_size=1, strides=1, padding='same')(output1_0)
    output1 = tf.keras.layers.add([shortcut1, output1_1])

    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2_0 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2_1 = convolution_block(first_layer_filter_count, output2_0)
    shortcut2 = tf.keras.layers.Conv2D(first_layer_filter_count, kernel_size=1, strides=1, padding='same')(output2_0)
    output2 = tf.keras.layers.add([shortcut2, output2_1])

    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3_0 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3_1 = convolution_block(first_layer_filter_count, output3_0)
    shortcut3 = tf.keras.layers.Conv2D(first_layer_filter_count, kernel_size=1, strides=1, padding='same')(output3_0)
    output3 = tf.keras.layers.add([shortcut3, output3_1])

    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                           outputs=[output1, output2, output3])

    return model


def make_mulmo_3encUNet_nested_decoder_no_cclat(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_m3 = base_model3.get_layer('block1_conv2_m3').output
    enc1_m3 = base_model3.get_layer('block2_conv2_m3').output
    enc2_m3 = base_model3.get_layer('block3_conv2_m3').output
    enc3_m3 = base_model3.get_layer('block4_conv3_m3').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output1)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output2)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output3)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m1])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m2])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3_m3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m1])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2_m3])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m2])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1_m3])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m1])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m2])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0_m3])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                           outputs=[output1, output2, output3])

    return model

def make_mulmo_3encUNet_nested_decoder_add(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    # vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2, lat_output3])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    enc0_m3 = base_model3.get_layer('block1_conv2_m3').output
    enc1_m3 = base_model3.get_layer('block2_conv2_m3').output
    enc2_m3 = base_model3.get_layer('block3_conv2_m3').output
    enc3_m3 = base_model3.get_layer('block4_conv3_m3').output

    first_layer_filter_count = 64


    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output1)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output2)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output3)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)
    dec3_add = tf.keras.layers.Add()([dec3_m1, dec3_m2, dec3_m3])
    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_add, enc3_m1])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_add, enc3_m2])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_add, enc3_m3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_add = tf.keras.layers.Add()([dec2_m1, dec2_m2, dec2_m3])
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_add, enc2_m1])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_add, enc2_m2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_add, enc2_m3])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_add = tf.keras.layers.Add()([dec1_m1, dec1_m2, dec1_m3])
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_add, enc1_m1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_add, enc1_m2])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_add, enc1_m3])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_add = tf.keras.layers.Add()([dec0_m1, dec0_m2, dec0_m3])
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_add, enc0_m1])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_add, enc0_m2])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_add, enc0_m3])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                           outputs=[output1, output2, output3])

    return model


def make_UNet_3_dec(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False
    lat_output = base_model.get_layer('block5_conv3').output

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0 = base_model.get_layer('block1_conv2').output
    enc1 = base_model.get_layer('block2_conv2').output
    enc2 = base_model.get_layer('block3_conv2').output
    enc3 = base_model.get_layer('block4_conv3').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, enc3])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, enc3])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m3, enc3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, enc2])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, enc2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m3, enc2])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, enc1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, enc1])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m3, enc1])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, enc0])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, enc0])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m3, enc0])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model.input],
                           outputs=[output1, output2, output3])

    return model


def make_UNet_3_nested_dec(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False
    lat_output = base_model.get_layer('block5_conv3').output

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0 = base_model.get_layer('block1_conv2').output
    enc1 = base_model.get_layer('block2_conv2').output
    enc2 = base_model.get_layer('block3_conv2').output
    enc3 = base_model.get_layer('block4_conv3').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, dec1_m2, dec1_m3, enc1])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m3, dec1_m2, dec1_m3, enc1])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, dec0_m2, dec0_m3, enc0])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m3, dec0_m2, dec0_m3, enc0])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model.input],
                           outputs=[output1, output2, output3])

    return model


def make_UNet_3_nested_dec2(height, width, channels, output_channels):
    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False
    lat_output = base_model.get_layer('block5_conv3').output

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0 = base_model.get_layer('block1_conv2').output
    enc1 = base_model.get_layer('block2_conv2').output
    enc2 = base_model.get_layer('block3_conv2').output
    enc3 = base_model.get_layer('block4_conv3').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(lat_output)
    dec3_m3 = tf.keras.layers.BatchNormalization()(dec3_m3)

    dec3_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, dec3_m2, dec3_m3, enc3])
    dec3_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, dec3_m3, enc3])
    dec3_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, dec3_m3, enc3])

    # (N/4 x N/4 x 4CH)
    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_concat_m1)
    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_concat_m2)
    dec2_m3 = decoding_block(first_layer_filter_count * 4, dec3_concat_m3)
    dec2_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, dec2_m2, dec2_m3, enc2])
    dec2_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, dec2_m3, enc2])
    dec2_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, dec2_m3, enc2])

    # (N/2 x N/2 x 2CH)
    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_concat_m1)
    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_concat_m2)
    dec1_m3 = decoding_block(first_layer_filter_count * 2, dec2_concat_m3)
    dec1_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, dec1_m2, dec1_m3, enc1])
    dec1_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, dec1_m3, enc1])
    dec1_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, dec1_m3, enc1])

    # (N x N x CH)
    dec0_m1 = decoding_block(first_layer_filter_count, dec1_concat_m1)
    dec0_m2 = decoding_block(first_layer_filter_count, dec1_concat_m2)
    dec0_m3 = decoding_block(first_layer_filter_count, dec1_concat_m3)
    dec0_concat_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, dec0_m2, dec0_m3, enc0])
    dec0_concat_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, dec0_m3, enc0])
    dec0_concat_m3 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, dec0_m3, enc0])

    # Last layer 1 : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_concat_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # Last layer 2 : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_concat_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    # Last layer 3 : CONV BLOCK + convolution + Sigmoid
    output3 = convolution_block(first_layer_filter_count, dec0_concat_m3)
    output3 = convolution_block(first_layer_filter_count, output3)
    output3 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output3)
    output3 = tf.keras.layers.Activation(activation='sigmoid')(output3)

    model = tf.keras.Model(inputs=[base_model.input],
                           outputs=[output1, output2, output3])

    return model



if __name__ == '__main__':
    model = make_UNet_3_nested_dec(224, 224, 3, 1)
    model.summary()