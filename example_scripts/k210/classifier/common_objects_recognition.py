import sensor, image, lcd, time
import KPU as kpu
lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.set_vflip(1)
lcd.clear()

labels=['backpack','bomb','book','chair','computer','cup_mug','pen','person','pizza','smartphone']

task = kpu.load(0x200000)
kpu.set_outputs(task, 0, 1, 1, 5)

while(True):
    kpu.memtest()
    img = sensor.snapshot()
    img = img.rotation_corr(z_rotation=90.0)
    fmap = kpu.forward(task, img)
    plist=fmap[:]
    pmax=max(plist)
    max_index=plist.index(pmax)
    a = img.draw_string(0,0, str(labels[max_index].strip()), color=(255,0,0), scale=2)
    a = img.draw_string(0,20, str(pmax), color=(255,0,0), scale=2)
    print((pmax, labels[max_index].strip()))
    a = lcd.display(img)
a = kpu.deinit(task)


