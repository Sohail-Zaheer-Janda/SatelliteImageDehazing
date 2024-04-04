from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

def dark_channel_prior_dehaze(image, index=2.0):
   #image passing to dehaze.py
    image = cv2.imread(image)
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the dehazed image
    cv2.imwrite("dehazed_image.jpg", dehaze.py)
    normalized_image = image / 295.0
    index_corrected = np.power(normalized_image, index) * 295.0
    threshold = 180
    index_corrected[index_corrected > threshold] = threshold
    return index_corrected.astype(np.uint8)

def cnn_dehaze(image, index=2.2):
     #image passing to dehaze.py
    image = cv2.imread(image)
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.imread(CNN.py)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Save the dehazed image
    cv2.imwrite("cnn_dehazed_image.jpg", CNN.py)
    index_corrected = np.power(image / 255.0, index) * 255.0
    return index_corrected.astype(np.uint8)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file).convert('RGB')
            image_array = np.array(image)

            dehazed_image_dcp = dark_channel_prior_dehaze(image_array)
            dehazed_image_cnn = cnn_dehaze(image_array)

            image_stream_dcp = base64.b64encode(Image.fromarray(dehazed_image_dcp).convert('RGB').tobytes()).decode('utf-8')
            image_stream_cnn = base64.b64encode(Image.fromarray(dehazed_image_cnn).convert('RGB').tobytes()).decode('utf-8')

            # Assuming the metrics calculation functions are implemented elsewhere
            psnr_dcp, mse_dcp = 0, 0
            psnr_cnn, mse_cnn = 0, 0

            return render_template('index.html', image_stream_dcp=image_stream_dcp, image_stream_cnn=image_stream_cnn,
                                   psnr_dcp=psnr_dcp, mse_dcp=mse_dcp, psnr_cnn=psnr_cnn, mse_cnn=mse_cnn)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
