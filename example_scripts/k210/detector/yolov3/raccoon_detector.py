# needs firmware from my fork with yolov3 support, see
# https://github.com/sipeed/MaixPy/pull/451

import sensor, image, lcd
import KPU as kpu

lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_vflip(1)
sensor.run(1)

classes = ["raccoon"]

task = kpu.load(0x300000) #change to "/sd/name_of_the_model_file.kmodel" if loading from SD card
a = kpu.set_outputs(task, 0, 10, 8, 18) #the actual shape needs to match the last layer shape of your model(before Reshape)
anchor = (0.76120044, 0.57155991, 0.6923348, 0.88535553, 0.47163042, 0.34163313)

a = kpu.init_yolo3(task, 0.5, 0.3, 3, 1, anchor) 
# second parameter - obj_threshold, tweak if you're getting too many false positives
# third parameter - nms_threshold
# fourth parameter - number of anchors
# fifth parameter - number of branches for YOLOv3, in this case we only use one branch

while(True):
    img = sensor.snapshot()
    #a = img.pix_to_ai() # only necessary if you do opeartions (e.g. resize) on image
    code = kpu.run_yolo3(task, img)

    if code:
        for i in code:
            a = img.draw_rectangle(i.rect(),color = (0, 255, 0))
            a = img.draw_string(i.x(), i.y(), classes[i.classid()], color=(255,0,0), scale = 1.5)
        a = lcd.display(img)
    else:
        a = lcd.display(img)
a = kpu.deinit(task)
