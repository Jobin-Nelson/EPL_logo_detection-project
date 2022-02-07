import os, glob, shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_data(data_path, train_path, val_path, test_path, split_size=0.3):
    folders = os.listdir(data_path)

    for folder in folders:
        full_path = os.path.join(data_path, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))
        x_train, val_test = train_test_split(images_paths, test_size=split_size)
        x_val, x_test = train_test_split(val_test, test_size=0.5)

        for x in x_train:
            new_folder = os.path.join(train_path, folder)

            if not os.path.isdir(new_folder):
                os.makedirs(new_folder)

            shutil.copy(x, new_folder)

        for x in x_val:
            new_folder = os.path.join(val_path, folder)

            if not os.path.isdir(new_folder):
                os.makedirs(new_folder)

            shutil.copy(x, new_folder)

        for x in x_test:
            new_folder = os.path.join(test_path, folder)

            if not os.path.isdir(new_folder):
                os.makedirs(new_folder)

            shutil.copy(x, new_folder)
        
def create_generators(batch_size, train_path, val_path, test_path):
    preprocessor = ImageDataGenerator(
        rescale= 1/255.
    )

    train_gen = preprocessor.flow_from_directory(
        train_path,
        class_mode='categorical',
        target_size=(140, 140),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )
    val_gen = preprocessor.flow_from_directory(
        val_path,
        class_mode='categorical',
        target_size=(140, 140),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )
    test_gen = preprocessor.flow_from_directory(
        test_path,
        class_mode='categorical',
        target_size=(140, 140),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )
    return train_gen, val_gen, test_gen

if __name__ == '__main__':
    data_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\epl-logos-big\\epl-logos-big'
    train_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\train'
    val_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\val'
    test_path = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\test'

    # split_data(data_path=data_path,
    #             train_path=train_path,
    #             val_path=val_path,
    #             test_path=test_path)