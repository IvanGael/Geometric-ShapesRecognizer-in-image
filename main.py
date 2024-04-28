from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

def detect_shapes(image_base64):
    # Convert base64 string to numpy array
    nparr = np.frombuffer(base64.b64decode(image_base64), np.uint8)
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate shape by a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Identify the geometric shape based on the number of sides
        num_sides = len(approx)
        shape = ""
        if num_sides == 3:
            shape = "Triangle"
        elif num_sides == 4:
            # Check if it's a square, rectangle, or diamond
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            elif 1.2 <= aspect_ratio <= 1.8:
                shape = "Rectangle"
            else:
                shape = "Diamond"
        elif num_sides == 5:
            shape = "Pentagon"
        elif num_sides == 6:
            # Check if it's a trapezoid
            shape = "Trapezoid"
        else:
            # Check if it's a circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            circle_area = np.pi * radius ** 2
            if abs(area - circle_area) < 200:  
                shape = "Circle"
            else:
                shape = "Unrecognized Shape"

        return shape

@app.route('/detect_shapes', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['file']
    image_base64 = base64.b64encode(image.read()).decode('utf-8')

    shape = detect_shapes(image_base64)

    return jsonify({'shape': shape}), 200

if __name__ == '__main__':
    app.run(debug=True)
