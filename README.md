# 🔬 Automated Multi-Stage Leukemia Classification (Hybrid Deep Learning System)

This repository contains the codebase and trained models for an intelligent, web-based system designed to classify the stage of leukemia using peripheral blood smear (PBS) images. We leveraged the synergy between Convolutional Neural Networks (CNNs) and temporal modeling architectures to achieve state-of-the-art diagnostic performance.

## 1. Project Objectives

The core objectives of this research and development project are:

*   **Multi-Stage Classification:** To accurately categorize PBS images into four distinct leukemia stages: **Benign, Early (Early Pre-B), Pre-leukemic (Pre-B), and Pro-leukemic (Pro-B)**.
*   **Reduce Subjectivity and Delay:** To provide medical professionals with a fast, accurate, and objective diagnostic tool to minimize the subjectivity and time associated with manual microscopy.
*   **Hybrid Modeling:** To develop powerful diagnostic models by combining advanced CNNs (for spatial features) with sequential models (Bi-GRU, LSTM, ViT) to extract both spatial and sequential patterns from the image data.
*   **Web Accessibility:** To deploy the system as a user-friendly, full-stack web application (using Python/Flask and HTML) that allows users to register, log in, upload images, and receive instant predictions.
*   **Benchmarking:** To rigorously benchmark four unique hybrid architectures to determine the most effective and robust model for clinical use.

## 2. Hybrid Deep Learning Models Used

We tested four hybrid deep learning models that integrate CNN feature extractors with sequential/temporal processing units:

| Hybrid Model Architecture | CNN (Spatial Feature Extraction) | Sequential/Temporal Component | Key Feature/Purpose |
| :--- | :--- | :--- | :--- |
| **Xception + Bi-GRU** | Xception (Depthwise Separable Convolutions) | Bidirectional Gated Recurrent Unit (Bi-GRU) | Captures fine-grained spatial characteristics and accounts for contextual dependencies (forward and backward). |
| **EfficientNetB3 + Bi-GRU** | EfficientNetB3 (Scalable & Efficient CNN) | Bidirectional Gated Recurrent Unit (Bi-GRU) | Balances efficiency with high accuracy, modeling sequential dependencies across spatial image regions. |
| **EfficientNetB3 + Vision Transformer (ViT)** | EfficientNetB3 (Efficient Feature Encoder) | Vision Transformer (ViT) | Combines efficient local feature extraction with ViT's ability to capture **global long-range dependencies** between image patches. |
| **MobileNetV3 + LSTM** | MobileNetV3 (Compact CNN for Edge Devices) | Long Short-Term Memory (LSTM) | Optimized for **fast inference** and resource efficiency, using LSTM to identify **temporal relations/latent patterns** in cell morphology progression. |

## 3. System Implementation Flow (Step-by-Step)

### A. Backend and Data Pipeline

1.  **Data Collection:** Acquired 3,256 high-resolution PBS images from a curated Kaggle dataset collected from Taleqani Hospital, Tehran, Iran. Images were labeled based on expert pathologist verification (ground truth).
2.  **Data Preprocessing and Balancing:** Applied resizing, normalization, and segmentation (where needed). Used **data augmentation** (rotation, flipping, contrast adjustment) to balance the original dataset, resulting in a total of **3,916 images** across the four classes.
3.  **Data Splitting:** The balanced dataset was partitioned using **stratified splitting** to ensure proportional class representation in all sets. The final test set contained **784 images**.
4.  **Model Training:** All four hybrid architectures were trained using optimized hyperparameters and regularization techniques to prevent overfitting.
5.  **Model Evaluation:** All models were evaluated using confusion matrices, accuracy, precision, recall, and F1-score on the independent test set of 784 images.
6.  **Model Prediction Deployment:** The **best-performing model** was integrated into the **Python/Flask backend** for real-time inference on user uploads.

### B. User Flow (Frontend)

1.  **Registration Page:** New users create accounts.
2.  **Login Page:** Registered users log in securely using Flask session management.
3.  **Image Upload & Prediction Page:** Users upload a PBS image via the simple HTML interface.
4.  **Prediction:** The backend classifies the image into one of the four stages and returns the result in real-time.
5.  **Logout:** Securely terminates the user session.

## 4. Key Results and Performance Metrics

The evaluation demonstrated that all hybrid models achieved high diagnostic accuracy (over 90%).

| Hybrid Model Architecture | Overall Accuracy (Test Set: 784 Images) | Macro Avg. F1-Score | Key Misclassifications |
| :--- | :--- | :--- | :--- |
| **MobileNetV3 + LSTM** | **1.00 (100%)** | **1.00** | **Zero Misclassifications** (Flawless) |
| EfficientNetB3 + Bi-GRU | 0.99 (99%) | 0.99 | Minimal errors; Class 3 (Pro-leukemic) was 100% accurate. |
| EfficientNetB3 + ViT | 0.99 (99%) | 0.99 | Minimal errors; Pre-leukemic achieved perfect precision (1.00). |
| Xception + Bi-GRU | 0.94 (94%) | 0.94 | Most confusion occurred between adjacent stages (Benign/Early). |

## 5. Chosen Model for Deployment: MobileNetV3 + LSTM

The **MobileNetV3 + LSTM** model was selected as the definitive architecture for deployment and integration into the final web system.

### Rationale for Selection:

*   **Perfect Classification Performance:** It achieved a **100% classification accuracy** on the 784 test images. The confusion matrix showed **zero misclassifications**.
*   **Flawless Metrics:** It delivered **perfect precision, recall, and F1-scores (1.00)** across all four distinct leukemia stages (Benign, Early, Pre-leukemic, Pro-leukemic).
*   **Efficiency for Clinical Settings:** MobileNetV3 is specifically designed to be a **compact CNN** optimized for **mobile and edge computing**. This makes the hybrid model highly suitable for deployment in clinical environments with restricted computing capabilities, offering fast and accurate real-time prediction.
*   **Robust Feature Fusion:** The combination leverages MobileNetV3 for efficient spatial feature extraction and LSTM for robust sequential modeling, enabling the detection of progressive changes and complex patterns of cell morphology indicative of leukemia progression.

## 6. Technical Stack

| Component | Technology | Rationale / Details |
| :--- | :--- | :--- |
| **Backend** | Python 3.8+ | Core programming language for data pipeline and deep learning. |
| **Web Framework** | Flask | Used for secure image processing, model inference, and handling user sessions/authentication. |
| **Deep Learning** | Pytorch, Timm | Framework used for model training, building architectures (Xception, EfficientNetB3, MobileNetV3), and inference. |
| **Frontend** | HTML, CSS, Bootstrap, JavaScript | Provides a simple, intuitive, and secure user interface for image uploads and result display. |
| **Database** | MySQL | Used for managing user credentials and session data. |
| **Hardware** (Minimum) | I3/Intel Processor, 8GB RAM | Reflects the system's design for operation on modest hardware in clinical settings. |

## 7. Future Enhancements

Potential future developments include:

*   **Explainable AI (XAI):** Integration of techniques (e.g., Grad-CAM) to provide visual interpretations of the model's decisions, boosting clinical trust and transparency.
*   **Cloud Deployment:** Utilizing cloud platforms (AWS/Azure) to enhance scalability and accessibility across multiple clinical centers.
*   **Multimodal Diagnosis:** Extending the system to integrate data from other modalities, such as bone marrow biopsy scans or flow cytometry reports.
*   **Active Learning:** Implementing a continuous feedback loop that uses expert re-annotation to refine the model over time, making it self-improving.
