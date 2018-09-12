from PIL import Image


def cut_image(img_name, img):
    #分割顺序为从上到下，从左到右
    weight, hight = img.size
    #1
    w = weight/2.0
    h = hight/2.0
    x = 0
    y = 0
    reigon = img.crop((x, y, x+w, y+h))
    reigon.save(img_name + '_1.jpg')
    #2
    x = w
    y = 0
    reigon = img.crop((x, y, x + w, y + h))
    reigon.save(img_name + '_2.jpg')
    #3
    x = 0
    y = h
    reigon = img.crop((x, y, x + w, y + h))
    reigon.save(img_name + '_3.jpg')
    #4
    x = w
    y = h
    reigon = img.crop((x, y, x + w, y + h))
    reigon.save(img_name + '_4.jpg')


image_num = int(input("输入图片数量："))          #指定需要裁剪图片的数量
print("当前指定图片数量：" + str(image_num))
image_path = input("输入图片路径：")     #指定图片所在目录
print("当前指定图片路径：" + image_path)
image_name = input("输入图片统一命名前缀：")
print("图片命名前缀为：" + image_name)
for i in range(image_num):
    name = image_name + '_' + str(i+1)
    im = Image.open(image_path+name+'.jpg')
    cut_image(image_path+name, im)

print("all done!")
