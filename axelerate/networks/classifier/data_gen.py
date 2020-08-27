from keras.preprocessing.image import ImageDataGenerator
from axelerate.networks.common_utils.augment import process_image_classification
import cv2


def preprocess(image, input_size, norm, augment):

    image = process_image_classification(image, input_size[0],input_size[1], augment)
    image = norm(image)
    return image

def create_datagen(train_folder, valid_folder, batch_size, input_size, project_folder, augumentation, norm):

    if not valid_folder:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess(input_size, norm, augumentation), validation_split=0.1)
        train_generator=train_datagen.flow_from_directory(train_folder,
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True,
				                                         subset='training')

        validation_generator=train_datagen.flow_from_directory(train_folder,
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True,
				                                         subset='validation')
    else:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess(input_size, norm, augumentation))
        train_generator=train_datagen.flow_from_directory(train_folder,
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True)

        validation_generator=train_datagen.flow_from_directory(valid_folder,
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True)		
				                                         
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    fo = open(os.path.join(project_folder,"labels.txt"), "w")
    for k,v in labels.items():
        print(v)
        fo.write(v+"\n")
    fo.close()
    return train_generator, validation_generator
