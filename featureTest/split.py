from PIL import Image

with Image.open('DIW057.tif') as img:
    for i in range(2):
        try:
            img.seek(i)
            img.save('Image_%s.tif'%(i,))
        except EOFError:
            break
