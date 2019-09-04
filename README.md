# WAIC2019 hackthon (Webank Garbage Classification Track)

## 1st Place Solution to WAIC2019 hackthon Garbage Classification Challenge

**Time: August 29-31, 2019, Shanghai, China**

**Team: Skye**

**Team Member: Skye (yeah, I'm a solo player)**

## 1 Intro
   This repo is the solution of team **Skye**, the hackthon lasts for **36** hours, all contestants 
   need to develop a model for garbage image classification within the specified time. See [slides]().
   
   Since July 1, 2019, Shanghai has taken the lead in implementing the 
   four-category policy of garbage (all garbage are classified into 4 categoires: harmful/recyclable/other/kitch). 
   Since my roommates and me are firm practitioners of this policy, I'm very interested in this challenge and 
   participated in alone (they are better at hardware than programming).
  
## 2 Rules
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
   
## 3 Method
### 3.1 Problem Analysis

  Before doing anything, I analyzed this problem and made 2 conclusions:
  
  **1. Level-2 label is more important than level-1 label**: Since a `mapping_list.txt` is provided, we can always get the right level-1 label by mapping level-2 to level-1.
  
  **2. Label Mapping is important for inference**: Although we have `mapping_list.txt`,  here is a question when directly mapping level-2 label to level-1 label: *Level-1 label or mapped level-2 label, which one should be trusted during inferencing?* It is obvious that setting proper confidence distribution for level-1 label and mapped level-2 label is rather important for inference.

  Based on the analysis, my strategy is consists of three parts: **classifier design**, **feature extractor design** and **inference strategy**.

### 3.2 Classifier Design
  
  I treat this problem as a **multi-class image classification** problem, each image has two classes so the multi-hot label 
  vector is filled with 2 ones and 401 zeros. (403 labels in total: 4 level-1 label + 399 level-2 label)
  
  So the classifier is simply a `Softmax` classifier with a fully connected layer, and loss is `multi-class cross entropy loss`.
  
  Another intesting design is utilizing two classifiers: one for level-1 label classification and the other for level-2. 
  However, after training my model with this kind of design, loss gets harder to convergence. (Maybe sharing the same feature 
  extractor parameter weights for both classifiers is wrong.)
  
### 3.3 Feature Extractor Design
  
  **Baseline**: SENet154 (Squeeze-and-Excitation Network)
  
   This is a strong baseline for ImageNet classification, I utilized this model and the weights pretrained on ImageNet, then fine-tuned all weights (with a modified last linear layer).
  
  **Local Feature**: Non-local Block 2D (Gaussian Version)
  
   Non-local block has a very nice property that the input size and output size are always equal, so it is rather easy to insert the block to any off-the-shelf model. I inserted three non-local blocks (see `pretrainedmodels/models/senet.py` for details.)
  
  **Global Feature**: NetVLAD Encoding
  
   NetVLAD is an effective feature aggregation method and is widely applied to image/video understanding challenges (see YouTube-8M Video Understanding Challenge.). I applied NetVLAD encoding to feature from the last average pooling layer of SENet154.
  
### 3.4 Inference Strategy(\*)
 
   Inference strategy including label mapping is very important as aforementioned. I came up with three kinds of inference strategies during competition:
   
   + **Naïve Inference**
   
    with torch.no_grad():
            batch_size = outputs.size(0)
            valid_indices = [0,1,2,3]
            valid_pred = torch.zeros(outputs.size())
            for i in range(batch_size):
                valid_pred[i][valid_indices] = outputs[i][valid_indices].cpu().float()
            _, pred_indices = valid_pred.topk(1,1,True,True)

   I utilized Naïve Inference for final submission, the Naïve Inference only takes indices `[0,1,2,3]` into account, the corresponding predictions are `valid_pred`, then `argmax(valid_pred)` is chosen as final prediction.

   + **Hard Mapping**
    
    with torch.no_grad():
            batch_size = outputs.size(0)
            _, pred = outputs.topk(topk, 1, True, True)
            for j in range(batch_size):
                hash_table = np.zeros([4])
                single_pred = pred[j]
                for i in range(len(single_pred)):
                    if single_pred[i].item() not in [0,1,2,3]:
                        hash_table[self.label_map[str(single_pred[i].item())]] += 1
                    else:
                        hash_table[single_pred[i].item()] += 1

                index = np.argmax(hash_table)

   Take the top k predictions as candidate perdictions, then if level-1 label is in top k predictions (denoted as `P_valid`), take `argmax(P_valid)`; if all top k predictions are level-2 label, then mapping level-2 label to level-1 label and vote for final prediction. (Maybe taking top 1 mapped level-2 label is a better choice.)
   
   + **Soft Mapping**
   
   Experimental results show that hard mapping result in clear performance drop, this is because that voting is not suitable for the classifier. (`for example, a prediction vector [0.8,0.5,0.5,0.5,0.5], the correct answer is 0, however when applying voting, the answer is not 0.`)
   
   A better design for mapping level-2 label to level-1 label is letting the mapping process to be learnt by model. For example, we can use two classifiers: level-1 classifier and level-2 classifier, the predictions from them are level-1 predictions (4-D vector) and level-2 predictions (399-D vector) respectively. Then we can use linear transformation to map a 399-D vector to a 4-D vector and use `torch.mm(mapped_vector, level-1 predictions)` to get final predictions, which is decided by both level-1 classifier and level-2 classifier.
   
## 4 Experiments

Training dataset: 50%

Evalution dataset: 50%

For inference: 95% training data

**Validation Results**

Model| Acc@1 | Acc@5 | Naïve Rank@1
-|-|-|-
SENet Finetune|83.207|96.755|86.325
Non-local SENet|84.677|96.875|87.601
Non-local SENet + NetVLAD Encoding|85.360|96.889|87.730


**Inference Results**

Model | Naïve Rank@1
-|-
SENet Finetune | 0.797
Non-local SENet + NetVLAD Encoding | 0.807

An interesting observation: testing results are much lower than validation results, that is because the training and validation data are crawled from web while the testing data is captured in our daily life which is the **real garbage**!

## Acknowlegements 

Many thanks to WAIC committe（世界人工智能大会）, Tencent Webank（腾讯微众银行） and Synced (机器之心).

## Contact

If you are interested in my project, for sharing solutions or discussing questions, please sent me an e-mail: [skyezx2018@gmail.com](skyezx2018@gmail.com)
