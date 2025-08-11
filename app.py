from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png","jpg","jpeg"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model artifacts
model = joblib.load("svm_plant_model.pkl")
scaler = joblib.load("svm_scaler.pkl")
le = joblib.load("label_encoder.pkl")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def extract_features_from_image(image_path):
    """
    Extract the same features as in the CSV.
    Returns: [mean_R, mean_G, mean_B, std_gray, edge_density, hue_mean, area_ratio]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read image.")

    # Resize to standard size
    img = cv2.resize(img, (256,256))
    # Convert color channels
    b,g,r = cv2.split(img)
    mean_B = float(b.mean())
    mean_G = float(g.mean())
    mean_R = float(r.mean())

    # Grayscale stats
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_gray = float(gray.std())

    # Edge density using Canny
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.sum()) / (edges.size * 255.0)  # normalized

    # Hue mean
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0].astype(float)
    hue_mean = float(hue.mean())

    # Simple area_ratio estimate (fraction of non-background pixels):
    # Use adaptive threshold and count non-zero
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,11,2)
    area_ratio = float(np.count_nonzero(th)) / (th.size)

    return [mean_R, mean_G, mean_B, std_gray, edge_density, hue_mean, area_ratio]

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    probs = None
    if request.method == "POST":
        # Option A: image upload
        if 'image' in request.files and request.files['image'].filename != '':
            img_file = request.files['image']
            if allowed_file(img_file.filename):
                filename = secure_filename(img_file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img_file.save(path)
                try:
                    feats = extract_features_from_image(path)
                except Exception as e:
                    return render_template("index.html", error=str(e))
                os.remove(path)  # cleanup
            else:
                return render_template("index.html", error="File type not allowed.")
        else:
            # Option B: manual numeric inputs fallback
            try:
                feats = [
                    float(request.form['mean_R']),
                    float(request.form['mean_G']),
                    float(request.form['mean_B']),
                    float(request.form['std_gray']),
                    float(request.form['edge_density']),
                    float(request.form['hue_mean']),
                    float(request.form['area_ratio'])
                ]
            except Exception as e:
                return render_template("index.html", error="Please upload an image or fill all numeric fields.")

        # scale and predict
        X = scaler.transform([feats])
        pred_idx = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(X)[0]
        # prepare readable probabilities
        prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}

        return render_template("index.html", prediction=pred_label, probabilities=prob_dict, features=feats)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
