# **Smart Surveillance System (SSS)**

## **Overview**
The **Smart Surveillance System (SSS)** is an advanced surveillance solution designed for university campuses. It integrates **facial recognition, behavior analysis, and real-time alerts** to detect and address security concerns such as **smoking** and **fighting** on campus. The system leverages **machine learning, computer vision, and real-time data analytics** to enhance campus safety and ensure compliance with institutional policies.

## **Features**
- **Live Video Feeds**: Real-time monitoring of campus activity.
- **Behavior Detection**: Automated detection of **smoking** and **fighting** incidents.
- **Facial Recognition**: Identifies individuals involved in violations.
- **Alert System**: Instant notifications for security personnel.
- **Data Logging and Reporting**: Stores incidents with timestamps and locations for analysis.
- **Web-Based Dashboard**: Centralized control panel for viewing and managing surveillance data.

---

## **System Setup**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/IAlphaK/Smart-Surveillance-System-SSS.git
```

### **Step 2: Open Project in Visual Studio Code**
Navigate to the `Web App` folder and open it in Visual Studio Code.

### **Step 3: Running the Server**
Install **Node.js 20.9.0** for a properly working web app without conflicts:
🔗 [Download Node.js 20.9.0](https://nodejs.org/en/blog/release/v20.9.0)

Do ensure that **Python** is installed, as some Node packages and third-party libraries like **Canvas.js** require Python for the Express.js backend.

```bash
cd "Web App"
cd server
npm install
npm start
```

### **Step 4: Running the Client**
```bash
cd "Web App"
cd client
npm install
npm start
```
---

## **Build MMAction2 from Source**

🔗 **Follow the official guide to build mmaction2 from source:** [Official MMAction2 Installation Guide](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html)

For compatibility, use the following versions:
- **CUDA 11.7**
- **PyTorch Installation:**
    ```bash
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
- **MMCV Installation:**
    ```bash
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
    ```

---

## **Dataset Details**
The dataset comprises two action classes: **Smoking** and **Fighting**.

- **Fighting Videos**: Sourced from a **public surveillance dataset** ([Fight Detection Dataset](https://github.com/seymanurakti/fight-detection-surv-dataset)).
- **Smoking Videos**: **Custom recorded** surveillance footage. **(Not shared due to privacy concerns).** 

### **Dataset Characteristics**
- **Total videos**: 108 (54 **Fighting** + 54 **Smoking**)
- **Training set**: 90 videos (45 per class)
- **Validation set**: 18 videos (9 per class)
- **Scene diversity**: Indoor and outdoor environments with varying camera angles and heights.
- **Average occupancy**: 4-5 people per frame with 1 target action performer.
- **Annotations**: Manually created using CVAT.

---

## **Model Configuration & Training**
### **Model Architecture**
The system uses the **SlowFast** network for **spatio-temporal action detection**.
- **Slow Pathway**: Captures detailed spatial semantics.
- **Fast Pathway**: Captures motion at high temporal resolution.

#### **ROI Head**
- Uses **AVARoIHead** with a **BBoxHeadAVA** to support multi-label classification.
- Includes a **background class** to improve detection accuracy.

### **Optimization Strategy**
- **Optimizer**: AdamW (`lr=0.0001`, `weight_decay=0.05`)
- **Learning Rate Scheduler**:
  - **Warm-up phase**: LinearLR for 5 epochs.
  - **Main phase**: CosineAnnealingLR (reduces LR gradually until epoch 40).

---

## **Running the Model for Inference**
### **Step 1: Install an IDE**
We recommend **PyCharm** for smooth execution.
🔗 [Install PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html)

### **Step 2: Run the Detection Model**
```bash
python main.py
```

### **Step 3: Adjust for CPU/GPU**
Modify the **"device" variable as cpu or cuda:0** based on whether you’re running the model on a **CPU** or **GPU**.

---

## **Results**
| Metric | Smoking | Fighting |
|---------|---------|---------|
| **Per-Class AP (Best Epoch)** | **57.08%** | **70.58%** |
| **Best Validation Accuracy** | **63.83% (at Epoch 7)** |

**Observations**:
- **Smoking detection (57.08% AP)**: The model had to detect subtle behaviors, which are more challenging compared to aggressive movements.
- **Fighting detection (70.58% AP)**: Higher accuracy due to more **dynamic motion patterns**.

---

## **Contact**
For inquiries or contributions, reach out to the **FYP Team**:

**Abdullah Basit**: [abdullahbasit017@gmail.com](mailto:abdullahbasit017@gmail.com)  
**Muhammad Abubakar Siddique**: [abubakarsidique694@gmail.com](mailto:abubakarsidique694@gmail.com)

**Supervised by**: **Dr. Usman Ghous**

---

## **Acknowledgment**
We extend our gratitude to **Seymanur Akti** for providing the **fight detection dataset**:  
🔗 **[Fight Detection Dataset](https://github.com/seymanurakti/fight-detection-surv-dataset)**

We also acknowledge **MMAction2** and **OpenMMLab** for their open-source contributions in video action recognition:  
**[MMAction2 Repository](https://github.com/open-mmlab/mmaction2)**  
**MMAction2 Contributors. OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark. 2020-07-21.**
