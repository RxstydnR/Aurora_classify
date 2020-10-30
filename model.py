

def get_model(_input_shape, num_classes, model_name):

    MODEL_NAME = model_name

    if MODEL_NAME=="CNN":
        from model import CNN
        model = CNN(_input_shape, num_classes)
        model.compile(loss = "categorical_crossentropy", # binary_crossentropy
                    optimizer="adam",
                    metrics=["accuracy"])
    
    elif MODEL_NAME=="ResNet50":
        from keras.applications.resnet50 import ResNet50

        model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=_input_shape, pooling=None, classes=num_classes)
        model.compile(loss = "categorical_crossentropy", # binary_crossentropy
                    optimizer="adam",
                    metrics=["accuracy"])

    elif MODEL_NAME=="MobileNetV2":    
        from keras.applications.mobilenet_v2 import MobileNetV2

        model = MobileNetV2(input_shape=_input_shape, alpha=1.0, include_top=True, weights=None, input_tensor=None, pooling=None, classes=num_classes)
        model.compile(loss = "categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    elif MODEL_NAME=="MobileNet":
        from keras.applications.mobilenet import MobileNet

        model = MobileNet(input_shape=_input_shape, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights=None, input_tensor=None, pooling=None, classes=num_classes)
        model.compile(loss = "categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    elif MODEL_NAME=="ResNet18":
        model = ResNet18(_input_shape, num_classes)
        model.compile(loss = "categorical_crossentropy", # binary_crossentropy
                    optimizer="adam",
                    metrics=["accuracy"])
    
    return model


def ResNet18(IMG_SHAPE, CLASSES):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Activation
    from keras.models import Model

    num_filters = 64
    num_blocks = 4
    num_sub_blocks = 2

    inputs = Input(shape=IMG_SHAPE)
    
    x = Conv2D(filters=num_filters, kernel_size=(7,7), padding='same', strides=2, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='block2_pool')(x)

    for i in range(num_blocks):
        for j in range(num_sub_blocks):
            
            strides=1
            
            is_first_layer_but_not_first_block=False
            if j==0 and i>0:
                is_first_layer_but_not_first_block=True
                strides=2

            y = Conv2D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
            y = BatchNormalization()(y)
            
            # Skip structure
            if is_first_layer_but_not_first_block:
                x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal')(x)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)

        num_filters *= 2

    x = GlobalAveragePooling2D()(x)
    
    outputs = Dense(CLASSES, activation='softmax')(x)#(flattened)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def CNN(IMG_SHAPE, CLASSES):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Activation
    from keras.models import Model

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same",
            kernel_initializer='he_normal',input_shape=IMG_SHAPE))
    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), strides=2,padding='same', activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(, CLASSES, activation='softmax')) # sigmoid

    return model