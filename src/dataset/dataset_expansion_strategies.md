# Dataset Expansion Strategies for Soccer Foul Detection

To significantly improve the Foul Detector model, increasing the dataset (especially the minority class of 934 "No Foul" clips) is the absolute best approach. 

Here are the most effective ways to increase your dataset for this specific soccer physics task:

### 1. Horizontal Flipping (Instant 2x Data)
In soccer, a tackle happening from the left side of the screen is physically identical to a tackle happening from the right side. 
*   **How to do it:** Take your raw videos, flip them horizontally (mirror them), and run your SigLIP feature extractor on the flipped videos. 
*   **The Result:** You instantly double your dataset! You will go from 934 "No Foul" clips to **1,868 "No Foul" clips** with zero manual labeling required.

### 2. Temporal Shifting (Time Jittering)
Currently, the model extracts exactly 16 frames for an action. 
*   **How to do it:** You can create "new" clips by shifting the extraction window slightly. For example, if your current clip extracts frames 10 to 26, create a new feature file that extracts frames 8 to 24, or 12 to 28. 
*   **The Result:** Because the Temporal Attention module receives a slightly different sequence, it acts as brand new training data and prevents temporal memorization. 

### 3. Hard Negative Mining (Manual Collection)
The model struggles because "clean tackles" look visually identical to fouls.
*   **How to do it:** Go to YouTube or soccer databases (like Wyscout) and search specifically for compilations titled *"Clean Tackles"*, *"Best Defensive Tackles"*, or *"Fair Play Defending"*. 
*   **The Result:** Manually clip 200-300 of these and add them to your dataset as `0` (No Foul). Adding specifically "hard" examples (where a player tackles aggressively but legally) will force the model to learn the fine-grained difference between touching the ball vs. touching the player's leg.

### 4. Pseudo-Labeling (Semi-Supervised)
*   **How to do it:** Take a long, unlabelled soccer match video. Run your current trained Foul Detector model (which currently has 58.6% balanced accuracy) on it. 
*   **The Result:** Have the model output a list of timestamps where it is **95% confident** that a "No Foul" is happening. You can then quickly watch those specific timestamps, confirm they are clean tackles, and instantly add them to your training set. This is much faster than watching a whole 90-minute game looking for tackles!
