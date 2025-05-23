# 🐘 Animal Detection and Counting in Aerial Images using HerdNet

This project implements a deep learning pipeline for **automatic detection, counting, and classification of African mammals** in aerial imagery. It is based on the **HerdNet architecture**, originally developed for wildlife detection in drone images. The system has been adapted and **fine-tuned on a custom dataset** to address specific challenges of species distribution, occlusion, and image resolution.

---

## 👨‍💻 Authors

**Alejandro Aristizábal**

**Alexander Hernández**

**Juan David Rico**

**Juan Felipe Jiménez**

Master’s Project – Universidad de los Andes, 2025

---

## 📁 Project Structure

```
.
├── Artículo Proy. Grado           # Academic report and annexes
├── Datos                          # Dataset (images, annotations, etc.)
├── Modelos                        # Trained HerdNet models (.pth)
├── Notebooks                      # Experimentation and fine-tuning notebooks
├── animaloc                       # Core codebase (data loading, models, training, evaluation)
├── tools                          # CLI tools for training, inference, visualization, etc.
├── app.py                         # Streamlit app for inference
├── requirements.txt              # Python dependencies
└── README.md                      # Project documentation
```

---

## 🧠 Abstract

Conflicts between wildlife and livestock in sub-Saharan Africa call for scalable biodiversity monitoring tools. This project develops and evaluates a deep learning-based system using the **HerdNet CNN model** to detect, count, and classify six mammal species from UAV imagery. The original bounding-box annotations were converted into point-based formats to improve performance under occlusion and crowding.

To adapt the model to the specific characteristics of the dataset, we performed **custom fine-tuning of HerdNet**, testing several strategies over selected layers and optimizing the model’s ability to generalize and classify animal species in high-resolution aerial environments.

---

## 🔧 Base Model: HerdNet

We used [**HerdNet**](https://github.com/Alexandre-Delplanque/HerdNet) as the foundational architecture. HerdNet is a convolutional neural network tailored for wildlife detection from UAV images using **point annotations** rather than bounding boxes. We:

* Integrated the original HerdNet codebase with our preprocessing and annotation pipeline.
* Applied **fine-tuning techniques** on different layers (deep and shallow) to adapt the model to our dataset.
* Customized learning rates, optimizers, and schedulers to improve training performance.

---

## 📊 Results Summary

| Experiment | F1 Score  | Recall    | Precision | MAE      | RMSE     |
| ---------- | --------- | --------- | --------- | -------- | -------- |
| Exp 1      | **79.91** | 74.11     | **86.69** | 0.59     | **1.12** |
| Exp 2      | 62.91     | 84.31     | 50.17     | 1.61     | 4.68     |
| Exp 3      | 68.82     | 82.25     | 59.16     | 1.39     | 2.65     |
| Exp 4      | 72.62     | **84.80** | 63.50     | **0.97** | 2.05     |

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Experiments

All training was conducted on Google Colab (A100 GPU). Fine-tuning strategies included:

1. **Baseline training** with pre-trained HerdNet weights
2. **Superficial fine-tuning** of classification layers
3. **Selective deep fine-tuning** of level4, level5, and fc layers
4. **Differentiated learning rates** with AdamW optimizer and ReduceLROnPlateau scheduler

All experiments were tracked using **MLflow** for reproducibility and comparison.

---

## 📂 Dataset

* Source: [University of Liège UAV imagery dataset](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)
* Species: Elephant, Buffalo, Kob, Warthog, Waterbuck, Topi
* Data split: Training, validation, test
* Each image was divided into `512x512` patches for efficient learning

---

## 🚀 Streamlit App

You can test the detection system with a public demo hosted on **Streamlit Community Cloud**:

🔗 **Live Demo:** [https://grupo5animaldetector.streamlit.app](https://grupo5animaldetector.streamlit.app)

Simply upload an aerial image and get automatic detection and counting results powered by our fine-tuned HerdNet model.

To run locally:

```bash
streamlit run app.py
```

---

## 🛠 CLI Tools

In `tools/`:

* `train.py`: Model training
* `infer.py`: Run model inference
* `patcher.py`: Image cropping
* `test.py`: Evaluation script
* `view.py`: Visualization utility

Example:

```bash
python tools/infer.py --model_path Modelos/herdnet_model_exp_4_OFFICIAL.pth --image_path Datos/test_10_pics_sample/example.jpg
```

---

## 📈 Future Directions

* Per-class data augmentation and synthetic data generation
* Training with multitask objectives (detection + classification + segmentation)
* Lighter backbones for real-time deployment
* Real-time integration via drones or remote monitoring systems

---

## 🔗 Useful Links

* 🧠 **HerdNet original repository**: [https://github.com/Alexandre-Delplanque/HerdNet](https://github.com/Alexandre-Delplanque/HerdNet)
* 📊 **UAV Dataset (University of Liège)**: [https://dataverse.uliege.be/file.xhtml?fileId=11098\&version=1.0](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)
* 🐾 **Streamlit Demo App**: [https://grupo5animaldetector.streamlit.app](https://grupo5animaldetector.streamlit.app)
* 📄 **Project Report**: See `Artículo_Grupo_5.pdf`
* 🔧 **Albumentations Library**: [https://albumentations.ai](https://albumentations.ai)
