from PIL import Image

img = Image.open('/home/sagar/Documents/codes/finalCodes/correlation/separteFrame/diw.tif')

for i in range(2):
    try:
        img.seek(i)
        img.save('Image_%s.tif'%(i,))
        
    except EOFError:
        break
