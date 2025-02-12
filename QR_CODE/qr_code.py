import qrcode
import matplotlib as plt
generate_image = qrcode.make("https://www.instagram.com/__aditya__soni_/")
generate_image.save("html.png")
print("genreted")