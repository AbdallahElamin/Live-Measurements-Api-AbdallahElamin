# AI-Based Human Body Measurement System for Tailoring & Fashion E-Commerce

This project is a real-time body measurement API built with **Flask**, **MediaPipe**, **OpenCV**, and **PyTorch**. By analyzing **front and side pose images** of a person, it calculates accurate human body measurements useful for tailoring, clothing size prediction, and virtual fitting rooms.

> 📸 Just send **front and side pose images** (captured using a smartphone or webcam) to this API, and receive key body measurements in centimeters — perfect for fashion retail platforms and tailor-made garment businesses.

---

## Features

- Real-time image-based body measurement
- AI-powered depth estimation using **MiDaS**
- Measurement accuracy with a deviation of **±2-3 cm**
- Calibrates scale using an **A4 paper** as a reference object
- Easily integratable into fashion e-commerce or tailoring platforms
- No external APIs — runs entirely on your local or server environment

---


## Libraries Used

| Library         | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `Flask`        | To expose a simple HTTP API                                             |
| `OpenCV`       | For image processing and contour detection                              |
| `MediaPipe`    | For pose landmark detection (shoulders, hips, etc.)                     |
| `PyTorch`      | For AI-based **depth estimation** using [MiDaS](https://github.com/isl-org/MiDaS) |
| `torchvision`  | Support for model loading & image transformations                       |

---

# What This Fork Changed
The single 458-line `app.py` has been split into a proper Python package structure:

```
Live-Measurements-Api-AbdallahElamin/
├── app.py              ← was 458 lines, now 13 lines
├── config.py           ← new: all constants
└── measurements/
    ├── __init__.py
    ├── depth.py        ← MiDaS model + estimate_depth()
    ├── calibration.py  ← focal length, scale factor, height-based distance
    ├── vision.py       ← contour-based body width scanner
    ├── calculator.py   ← all body measurement math
    ├── validator.py    ← front-image validation
    └── routes.py       ← Flask Blueprint + /upload_images route
```

### What Was Improved (Beyond Just Splitting)

- **Eliminated copy-pasted depth sampling**: The `_depth_ratio_at()` helper in `calculator.py` replaces 4 identical blocks that were copy-pasted across the chest, waist, hip, and thigh measurement sections.
- **Clear singleton pattern**: Both the MiDaS model (`depth.py`) and the MediaPipe Holistic instance (`routes.py`) are loaded once at module import — clearly documented, not hidden in a module body.
- **Flask Blueprint**: The route is now registered via app.register_blueprint(bp) — a proper Flask pattern that allows routes to be tested independently of the app.

---

# How It Works

1. Detects key landmarks using **MediaPipe Pose** (shoulders, hips, knees, ankles).
2. Uses **A4 paper** in the image to calibrate real-world scale from pixels.
3. Enhances width and depth estimation using the **MiDaS depth AI model**.
4. Calculates measurements using geometric approximations (**elliptical body model**).
5. Returns measurement data in **JSON format**.


## How to Run

After cloning the repository, it's highly important to note that this project is best run using `Python 3.12` so make sure you have `Python 3.12` and `Python Launcher` installed on your system.

First, `cd` into the project directory.
```bash
cd ..\Live-Measurements-Api-AbdallahElamin
```

Then, create a virtual environment and activate it.
```bash
py -3.12 -m venv .venv
```
```bash
.\.venv\Scripts\activate
```

Then, install the required dependencies.
```bash
pip install -r requirements.txt
```

Finally, run the following commands:
- Start the server:
```bash
.\.venv\Scripts\python.exe app.py
```
- Run the result script:
```bash
.\.venv\Scripts\python.exe result.py
```


# API Endpoint

**POST** `/measurements`

> ℹ️ For reference, see the images placed  in the root directory.

---
##  Request
Send a `multipart/form-data` **POST** request with the following fields:

- **`front_image`**: JPEG/PNG image captured from the front *(required)*
- **`side_image`** *(optional)*: JPEG/PNG image from the side *(for better accuracy)*
- **`user_height_cm`** : Real height of the person (in cm) for more precise calibration

---

###  Example using `curl`

```bash
curl -X POST http://localhost:5000/measurements \
  -F "front_image=@front.jpg" \
  -F "side_image=@side.jpg" \
  -F "user_height_cm=170"
```

# Measurements Provided

| **Measurement Name**     | **Description**                                                   |
|--------------------------|-------------------------------------------------------------------|
| `shoulder_width`         | Distance between left and right shoulders                        |
| `chest_width`            | Width at chest level                                              |
| `chest_circumference`    | Estimated chest circumference                                     |
| `waist_width`            | Width at waist level                                              |
| `waist`                  | Estimated waist circumference                                     |
| `hip_width`              | Distance between left and right hips                             |
| `hip_circumference`      | Estimated hip circumference *(if side image is given)*           |

---

> 📌 **Note:**  
- The system uses **AI depth maps** and **contour-based width detection**.  
- Final measurements may have a **±2–3 cm variance** depending on image quality and user alignment.
- Hip reading may be distorted by clothing, stance, or camera angle. Stand straight, keep feet under hips, arms slightly out, and avoid jackets or loose fabric.
- Waist reading may be too small. Make sure your midsection is fully visible and not heavily shadowed or blocked by arms.


# Integration in Fashion E-Commerce

This solution is plug-and-play for:

- **E-commerce brands** offering size suggestions or virtual try-ons.
- **Tailoring platforms** wanting remote client measurements.
- **Clothing manufacturers** personalizing size charts for customers.
- **Fashion mobile apps** for custom-fitted clothing suggestions.

Simply integrate this API into your frontend — mobile or web — to collect two photos and retrieve exact measurements.


## 🤝 Contributions

PRs and suggestions are welcome! Raise an issue, or open a pull request.

## 📜 License

MIT License. This software is originally developed by 
[JavTahir](https://github.com/JavTahir). This is merely a forked repository of the original project.