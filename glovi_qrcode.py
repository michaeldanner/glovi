import qrcode

# Data to be encoded
data = "https://www.visir.org/glovi-fragebogen/"

# Creating an instance of QRCode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# Adding data to the instance
qr.add_data(data)
qr.make(fit=True)

# Creating an image from the QR Code instance
img = qr.make_image(fill='black', back_color='white')

# Save the image to a file
img_path = "./GloVi_QR_Code.png"
img.save(img_path)

img_path
