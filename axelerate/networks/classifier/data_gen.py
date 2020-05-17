from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
import os
import cv2
import glob
import random 

def create_datagen(train_folder, valid_folder, batch_size, input_size, project_folder, augumentation):
    if augumentation:
        data_gen_args = dict(brightness_range=[0.5,1.5],
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    else:
        data_gen_args = {}


    
    if not valid_folder:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1, **data_gen_args)
        train_generator=train_datagen.flow_from_directory(train_folder,
                                                         target_size=(input_size[0],input_size[1]),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True,
				                                         subset='training')

        validation_generator=train_datagen.flow_from_directory(train_folder,
                                                         target_size=(input_size[0],input_size[1]),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True,
				                                         subset='validation')
    else:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, **data_gen_args)
        train_generator=train_datagen.flow_from_directory(train_folder,
                                                         target_size=(input_size[0],input_size[1]),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True)

        validation_generator=train_datagen.flow_from_directory(valid_folder,
                                                         target_size=(input_size[0],input_size[1]),
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
				                                         
def visualize_dataset(folder, num_imgs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_files_list = glob.glob(folder + '/**/*.jpg', recursive=True)
    random.shuffle(image_files_list)
    for filename in image_files_list[0:num_imgs]:
        image = cv2.imread(filename)
        cv2.putText(image, os.path.dirname(filename).split('/')[-1], (10,30), font, image.shape[1]/700 , (0, 0, 255), 2, True)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        plt.figure()
        plt.imshow(image)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        print(filename)
	
	                              
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str)
    parser.add_argument("--num_imgs", type=int)
    args = parser.parse_args()
    visualize_dataset(args.images, args.num_imgs)
