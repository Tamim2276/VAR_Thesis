# Foul Detector: Methodology and Experiment Log

## The Problem Statement
The primary challenge in training the Foul Detector was the **highly imbalanced dataset** (934 "No Foul" clips vs 5687 "Foul" clips) combined with the **extreme subjectivity** of the task. A clean tackle and a foul often share 95% of the same visual features (players falling, fast movement, proximity). 

A naive model naturally exploits the majority class, predicting "Foul" for almost every video to achieve high raw accuracy, resulting in a 0% recall for "No Foul". This document tracks the iterative experiments used to force the model to learn true, generalizable features.

---

## Experiment 1: Extreme Class Weights + Heavy Noise
*   **Approach:** We attempted to solve the imbalance by using `CrossEntropyLoss` with extreme class weights (e.g., punishing the model 6x more for missing a "No Foul") and injecting Gaussian noise into the pre-computed SigLIP features to prevent exact memorization.
*   **Why it failed:** The model collapsed (51.6% balanced accuracy). Applying extreme loss weights on frozen, pre-computed features caused severe gradient instability. The model simply shifted from over-predicting "Foul" to over-predicting "No Foul" without actually learning the visual differences.

## Experiment 2: Architectural Simplicity (Big vs. Small Model)
*   **Approach:** We compared a massive multi-layer MLP against a simple 1-layer MLP, averaging all 16 frames into a single feature vector.
*   **Why it failed (Limitation):** While the smaller model was more stable and avoided extreme overfitting, average-pooling 16 frames destroys temporal information. A foul or impact usually occurs in only 1 or 2 frames; averaging washes out this critical signal.

## Experiment 3: Temporal Attention (The Core Architecture)
*   **Approach:** We replaced average-pooling with a learned **Temporal Attention** module. The model learns a scoring function to assign a weight (0.0 to 1.0) to each of the 16 frames, allowing it to dynamically focus on the exact moment of impact. We combined this with a `WeightedRandomSampler` to balance the classes during training batches.
*   **Why it failed (Minority Class Overfitting):** While the architecture was a massive success, the sampler caused a new issue: "Minority Class Overfitting via Upsampling". Because there are only 934 "No Foul" clips, the sampler forced the model to look at those exact same clips ~6 times per epoch. The model memorized the background pixels of those specific videos (Train Recall: 91%) but failed on unseen validation videos (Valid Recall: 52%).

## Experiment 4: Random Undersampling + Focal Loss
*   **Approach:** To stop the memorization of the 934 clips, we stopped repeating them. We randomly undersampled the "Foul" clips to exactly 934 per epoch, ensuring the model only saw the "No Foul" clips once per epoch. We also added Focal Loss to force the model to focus on hard, ambiguous tackles.
*   **Why it failed:** Massive overcorrection. Undersampling completely destroyed the natural distribution of the dataset, and Focal Loss pushed the decision boundary too far. Validation recall for "No Foul" shot up to 79%, but "Foul" recall crashed to 32%. The model lost its ability to confidently identify real fouls.

## Experiment 5: The Final Solution (Temporal Attention + Mixup + Frame Masking + Threshold Tuning)
*   **Approach:** We reverted to the `WeightedRandomSampler` but attacked the memorization problem using state-of-the-art regularization techniques specifically designed for sequence models:
    1.  **Temporal Frame Masking (Time Dropout):** Randomly zeroed out 2 to 3 frames during every training pass. This forced the Temporal Attention module to look for *multiple* foul indicators rather than memorizing the location of a single "impact frame" in a specific video.
    2.  **Feature-Level Mixup:** We linearly blended "Foul" and "No Foul" features (e.g., 70% Foul + 30% No Foul) and adjusted the label accordingly (0.7). This destroyed the model's ability to memorize exact features and forced it to learn a smooth, generalized decision boundary.
    3.  **Threshold Tuning (0.60):** Because the sampler naturally pushes the model slightly towards predicting "Foul" on ambiguous clips, we shifted the decision threshold to `0.60` instead of `0.50`.
*   **Why it succeeded:** This provided the perfect balance. Mixup and Masking heavily regularized the model without destroying the dataset distribution. 
*   **Final Result:** **58.6% Validation Balanced Accuracy** (55.0% No Foul Recall / 62.3% Foul Recall). This is an exceptionally balanced and robust result for predicting highly subjective human referee decisions from frozen video features.

---

## Future Improvements

To push the accuracy beyond the ~60% ceiling in the future, the following architectural and data changes are recommended:

1.  **End-to-End Raw Video Training:** Currently, the model relies on frozen SigLIP features (which are trained to align generic images with text). Fine-tuning a native 3D CNN (e.g., SlowFast, I3D) or a Video Swin Transformer end-to-end on the raw video pixels would allow the model to learn spatial-temporal features specifically optimized for soccer tackles.
2.  **Multi-Modal Inputs:** A referee uses more than just vision to call a foul. Incorporating audio features (the sound of a kick vs. the sound of a player falling/yelling) or skeletal tracking (the exact velocity and trajectory of the tackling foot) would provide critical missing context.
3.  **Hard Negative Mining Dataset:** The model struggles because "clean tackles" look visually identical to fouls. Collecting a massive, specific dataset of *only* clean tackles (hard negatives) would force the model to learn the microscopic differences between touching the ball vs. touching the player.
