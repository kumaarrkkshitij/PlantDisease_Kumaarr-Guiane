import os
from flask import Flask, request, render_template
from src.predict_disease import predict_disease

UPLOAD_PATH = "static/uploads"
os.makedirs(UPLOAD_PATH, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict
            predicted_class, confidence = predict_disease(file_path)
            result = f"{predicted_class} ({confidence:.2f}%)"

            # Pass only filename to template
            img_filename = file.filename

    return render_template("index.html", result=result, img_filename=img_filename)

if __name__ == "__main__":
    app.run(debug=True)