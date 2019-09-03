# WAIC2019 hackthon (Webank Garbage Classification Track)

## 1st Place Solution to WAIC2019 hackthon Garbage Classification Challenge

**Time: August 29-31, 2019, Shanghai, China**

**Team: Skye**

**Team Member: Skye (yeah, I'm a solo player)**

## 1. Intro
   This repo is the solution of team **Skye**, the hackthon lasts for **36** hours, all contestants 
   need to develop a model for garbage image classification within the specified time.
   
   Since July 1, 2019, Shanghai has taken the lead in implementing the 
   four-category policy of garbage (all garbage are classified into 4 categoires: harmful/recyclable/other/kitch). 
   Since my roommates and me are firm practitioners of this policy, I'm very interested in this challenge and 
   participated in alone (they are better at hardware than programming).
  
## 2. Rules
   + **Datasets**
   
   Training set: ~20,000 images
   
   Testing set: 9,000 images (invisible to participants)
   
   + **Label**
   
   Every sample has two level labels: level-1 label for 4 coarse categories and level-2 for specific objects. For example:
   ```shell
   CD、DVD,2,223
   ```
   `2`: `other` 
   
   `223`: `CD、DVD`
   
   + **Evaluation Metric**
   
   The final accuracy only computed on 4 coarse categories (level-2 label is not required for inference), each sample has 
   **only one** category.
   
## 3. Method
  + **Problem Analysis**

  Before doing anything, I analyzed this problem and made 2 conclusions:
  
  **1. Level-2 label is more important than level-1 label**: Since a `mapping_list.txt` is provided, we can always get the right level-1 label by mapping level-2 to level-1.
  
  **2. Label Mapping is important for inference**: Although we have `mapping_list.txt`,  here is a question when directly mapping level-2 label to level-1 label: *Level-1 label or mapped level-2 label, which one should be trusted during inferencing?*. It is obvious that setting proper confidence distribution for level-1 label and mapped level-2 label is rather important for inference.

  + **Classifier Design**
  
  I treat this problem as a **multi-class image classification** problem, each image has two classes so the multi-hot label 
  vector is filled with 2 ones and 401 zeros. (403 labels in total: 4 level-1 label + 399 level-2 label)
  
  So the classifier is simply a `Softmax` classifier with a fully connected layer, and loss is `multi-class cross entropy loss`.
  
  Another intesting design is utilizing two classifiers: one for level-1 label classification and the other for level-2. 
  However, after training my model with this kind of design, loss gets harder to convergence. (Maybe sharing the same feature 
  extractor parameter weights for both classifiers is wrong.)
  
  + **Feature Extractor Design**
  
  **Baseline**: SENet154 (Squeeze-and-Excitation Network)
  
  **Local Feature**: Non-local Block 2D (Gaussian Version)
  
  **Global Feature**: NetVLAD Encoding
  
  
