# 🌿 FlameLeaf 🌿

**FlameLeaf** is a dual-model machine learning system designed to predict the fire risk of plants based on species recognition from leaf images. It combines:

- A **MobileNetV2-based deep learning model** for identifying plant species from uploaded leaf images.
- A **regression-based model** trained on the **FLAMITS dataset** to predict flammability-related traits and provide an overall fire risk rating (High / Medium / Low).

This project has potential applications in wildfire prevention, ecological monitoring, and forest resource management.

---

## Features

- 🔍 **Image-based species classification** using MobileNet.
- 🔥 **Fire risk prediction** using real-world flammability data.
- 📊 **Post-processing logic** to categorize raw numeric predictions into qualitative ratings.
- 🌐 **Flask API** to connect ML models with frontend.
- 🖼️ **Simple web frontend** for user interaction and image uploads.

---

## Tech Stack

- Python 3
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn 
- Flask (Backend API)
- HTML/CSS/JS (Frontend)

---

