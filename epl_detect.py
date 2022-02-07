import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Model
from utils import create_generators
import json

def model_1(n_classes):
    my_input = Input(shape=(140, 140, 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

if __name__ == '__main__':
    train_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\train'
    val_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\val'
    test_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\test'
    model_path = 'C:\\Users\\Jobin\\Ds\\My_projects\\EPL_logo_detection\\Models'

    batch_size = 64
    epochs = 10

    train_gen, val_gen, test_gen = create_generators(batch_size, train_path, val_path, test_path)
    n_classes = train_gen.num_classes
    # label_map = train_gen.class_indices
    # with open('labels.json', 'w') as f:
    #     json.dump(label_map, f)

    TRAIN_SAVE = False
    TRAIN = False
    TEST = False

    if TRAIN_SAVE:
        ckpt_saver = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor='val_accuracy', patience=7)

        model = model_1(n_classes)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            train_gen,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[ckpt_saver, early_stop]
        )

    if TRAIN:
        pass
        
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print('Evaluating validation set:')
        model.evaluate(val_gen)

        print('Evaluating test set:')
        model.evaluate(test_gen)
