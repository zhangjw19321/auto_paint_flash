from PIL import Image

def covert_transparent2black(image_path):
    try:
        imagePtah = image_path
        img = Image.open(imagePtah)
        if img.mode != 'RGBA':
            image = img.convert('RGBA')
        width = img.width
        height = img.height
        image = Image.new('RGB', size=(width, height), color=(0, 0, 0))
        image.paste(img, (0, 0), mask=img)
        image.save("image_bear1.png")        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    image_path = "f1.png"
    covert_transparent2black(image_path)
