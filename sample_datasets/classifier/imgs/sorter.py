import os, glob

image_files_list = glob.glob('.' + '/**/*.jpg', recursive=True)
i = 0
for file in image_files_list:
    i = i + 1
    folder_name = os.path.dirname(file).split('/')[-1]
    file_ext = file.split('.')[-1]
    os.replace(file,os.path.dirname(os.path.abspath(file))+'/'+folder_name+'_'+str(i)+'.'+file_ext)
    print(os.path.dirname(os.path.abspath(file))+'/'+folder_name+'_'+str(i)+'.'+file_ext)


