# AIRL Coding Assignments

This repository contains the documentation and implementation for two coding assignments:

1. **Vision Transformer (ViT) implementation on CIFAR-10**
2. **Text-driven Image & Video Segmentation pipeline using GroundingDINO and SAM 2**

---

## Assignment 1: Vision Transformer on CIFAR-10 (PyTorch)

This project implements a Vision Transformer (ViT) from scratch in PyTorch and trains it on the CIFAR-10 dataset. The objective is to achieve the highest possible test accuracy by experimenting with improvements and training techniques.

### How to Run in Colab

1. **Open Google Colab**
   Navigate to [Google Colab](https://colab.research.google.com).

2. **Upload Notebook**

   * Go to `File -> Upload notebook`
   * Select the `q1.ipynb` file from your local machine.

3. **Set Runtime Type**

   * Click `Runtime -> Change runtime type`
   * Select **GPU** from the *Hardware accelerator* dropdown menu and save.

4. **Run All Cells**

   * Click `Runtime -> Run all`
   * This installs dependencies, downloads CIFAR-10, defines the ViT model, and starts training.

---

### Best Model Configuration

| **Hyperparameter** | **Value** |
| ------------------ | --------- |
| Batch Size         | 128       |
| Learning Rate      | 3e-4      |
| Patch Size         | 4         |
| Stride             | 2         |
| Image Size         | 32        |
| Embedding Dim      | 256       |
| Number of Heads    | 8         |
| Transformer Depth  | 6         |
| MLP Dimension      | 512       |
| Dropout Rate       | 0.1       |
| Optimizer          | Adam      |

---

### Results

| **Experiment**                                      | **Test Accuracy (%)** |
| --------------------------------------------------- | --------------------- |
| Baseline (Non-Overlapping Patches, No Augmentation) | 65.48                 |
| With Data Augmentation                              | 69.14                 |
| Augmentation + AdamW + Scheduler                    | 69.28                 |
| Augmentation + Overlapping Patches + Adam (Best)    | 76.27                 |

---

### Bonus Analysis

**Data Augmentation Effects**

* Applying `RandomCrop`, `RandomHorizontalFlip`, and `ColorJitter` increased test accuracy from **65.48% → 69.14%**.
* Augmentation improves generalization by exposing the model to varied images, reducing overfitting.

**Optimizer and Scheduler Variants**

* Switching from **Adam** to **AdamW + CosineAnnealingLR** provided a slight boost (**69.14% → 69.28%**).
* AdamW improves generalization by decoupling weight decay.
* Cosine annealing gradually lowers learning rate, stabilizing convergence.

**Overlapping vs. Non-Overlapping Patches**

* Using overlapping patches (Patch Size = 4, Stride = 2) improved accuracy from **69.14% → 76.27%**.
* Overlapping patches allow the model to capture finer local context, leading to richer representations.

---

## Assignment 2: Text-Driven Image & Video Segmentation with SAM 2

This project demonstrates a powerful pipeline for performing segmentation on both static images and video clips using natural language text prompts. The system leverages the zero-shot object detection capabilities of GroundingDINO to interpret text and locate objects, and the high-quality segmentation power of the Segment Anything Model 2 (SAM 2) to generate precise masks.

### Core Technologies

* **GroundingDINO**: A state-of-the-art, open-set object detector that can locate arbitrary objects in an image based on a free-text query. It acts as the "eyes" of our pipeline, identifying where the object of interest is.
* **Segment Anything Model 2 (SAM 2)**: The successor to the original SAM, SAM 2 is a foundation model for image segmentation. It can generate high-quality masks for objects given various prompts, including points, boxes, or even other masks. In this pipeline, it takes the bounding boxes from GroundingDINO to determine the exact pixel-level boundaries of the object.

---

### ⚙️ The Pipeline: From Text to Mask

#### 1. Image Segmentation

* **Input**: Image + text prompt (e.g., *"the black cat sitting on the couch"*).
* **Region Seeding (GroundingDINO)**: Generates bounding boxes for objects matching the text.
* **Preprocessing for SAM 2**: Converts bounding boxes into the required format and prepares the image.
* **Mask Generation (SAM 2)**: Produces precise pixel-level masks.
* **Final Output**: Segmented object mask overlayed on the original image.

This is an end-to-end process requiring no manual annotation or retraining.

#### 2. Video Segmentation (Bonus Extension)

* **Initialization (First Frame)**: GroundingDINO + SAM 2 generate the first mask.
* **Propagation (Next Frames)**: Previous frame mask guides segmentation in the next frame.
* **Output**: Final segmented video with continuous object tracking.

---

### ⚠️ Limitations and Considerations

1. **Dependency on Initial Detection** – If GroundingDINO fails, SAM 2 masks will be poor.
2. **Simple Tracking Logic** – Fails under fast motion, occlusion, or multiple similar objects.
3. **No Re-identification** – Once the object is lost, it cannot be recovered.
4. **High Resource Demand** – Both models are computationally heavy and require GPU support for practical performance.

---

## Technologies Used

* **PyTorch**
* **GroundingDINO**
* **Segment Anything Model 2 (SAM 2)**
* **Google Colab (GPU runtime)**

---

## Authors

This work was completed as part of the **AIRL Coding Assignments**.
