# Stock Sentimental Prediction based on SEP

#### 🔎Introduction

"Summarize-Explain-Predict" [(SEP)](https://github.com/koa-fin/sep) model from the paper ["Learning to Generate Explainable Stock Predictions"](https://arxiv.org/abs/2402.03659) offers a framework for processing collections of text data and making binary stock price predictions (positive/ negative) with explanations. It uses a general purpose large LLM model (ChatGPT) in the cloud to train a local language model for the specific task of stock price prediction. This framework demonstrates the power of language model in capturing the abstract sentiment and stock price implication from text data, which can be extended to more extensive use in financial market. The explanations it provides help human analysts justify predictions and gain new insights during analysis. Detailed logic and explanations on how the SEP model works are provided in the "Project Description" section below.

#### 💡Improvements made in this repo

The SEP model originally uses tweets about targeted companies from Twitter as input. However, since X.com (Twitter) made significant changes to its API and blocked third-party scraping tools like Snscrape, scraping tweets and fetching search results has become much more difficult. To address this, I wrote a data collection script (`collect_data.ipynb`) that gathers news headlines from Google Finance News and uses them as text input for the SEP model. These news headlines serve a similar role to tweets but are much more readily available. The script can be easily extended to collect other types of textual data by following the format in `collect_data.ipynb`. This not only fills a gap in the official SEP repository, which does not provide data collection code, but also makes the framework more generalizable to various kinds of text data.

I reorganized the SEP training and inference (evaluation) code (in `train_customized_data.ipynb` and `evaluate_customized_data.ipynb`), making it much easier to train and utilize a custom language model. The current setup uses the `vicuna-7b-v1.5-16k model`, but it is straightforward to replace it with another model in the reorganized code. Additionally, I updated the API calls related to the OPENAI module (as of August 2025) to ensure the pipeline runs smoothly (the codes in the official SEP repo are already deprecated). It is also feasible to switch to other open-source LLMs instead of the costly OPENAI API by simply modifying a few lines in the reorganized code. Some other improvements are suggested at the end of this README.  

## 📜Usage

### Directory Structure
```
├── # Main codes (in jupyter notebook format)
├── collect_data.ipynb # main code to collect (1) Stock Price info & (2) News headlines
├── train_customized_data.ipynb # code to train the SEP model (OPENAI api key needed in training)
├── evaluate_customized_data.ipynb # code to evaluate the trained model or for inference
│ 
├── data # for storing data collected in collect_data.ipynb
│   ├── price # price data collected
│   └── tweet # text data (e.g. News, Tweeter) collected
├── datasets # for storing comparison data output in stage 1 of train_customized_data.ipynb
├── Llama_models # storing pretrained llama models
│   └── vicuna-7b-v1.5-16k # an example directory storing Llama models file
├── saved_models # storing the trained models
├── results # inference/ evaluation results
│ 
│ 
├── # Codes from official SEP repo
├── data_load
├── evaliate-main
├── explain_module
├── summarize_module
└── utils
```
- In the `data` directory, some data of NVDA from 2025 July 16 to 24 are present in the `price` and `tweet` directories as examples. 

- The Llama model used as local language model can be put in `Llama_models` directory. For example,`vicuna-7b-v1.5-16k model` files can be download from [here](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k/tree/main) into the `Llama_models/vicuna-7b-v1.5-16k` directory. In principle, the larger the Llama model you use, the better the performance. 

- The `saved_models` directory will be created after running the `train_customized_data.ipynb` notebook and the output models will be stored there. 

- Some sample evaluation result files are present in the `results` directory as examples.

### Enviornment Set-Up

Cuda 11.8 is required. This repo can only run in linux system. Other environmental requirements can be set up as below. 

```bash
# Conda Environment Setup (more recommended)
conda env create -f environment.yml

#OR by pip
pip install -r requirements.txt
```

#### Data collections

After setting up the environment as suggested in the "Enviornment Set-Up" section above, you can run the `collect_data.ipynb` to collect stock price data and News headline data for desired stock in a specified date range. The data collected will be stored in the data directory. 

#### Training and Inference (Evaluation)

You can run the `train_customized_data.ipynb` notebook for training. You will need to enter your API key in the ipynb notebook in order to use the OPENAI api. The trained models will be saved in the `saved_models` directory. 

For inference/ evaluation of the trained models, please run the `evaluate_customized_data.ipynb` notebooks. The results will be saved in the `results` directory (both the prediction label and the human readable explanation on the predicted labels).


## 📜Project Description 

## Major Ideas of SEP model

### (A) Training Logics in SEP and illustrations with a simple example

Example text input - 5 days of tweets on Apple Inc. (Xₜ⁻₅ to Xₜ⁻₁): <br />

Day -5: “Apple’s Q2 earnings beat analyst expectations.” <br />
Day -4: “iPhone 16 preorder numbers are record-breaking.” <br />
Day -3: “Regulatory concerns over App Store policies in EU.” <br />
Day -2: “Apple CEO expresses concerns over slowing China sales.” <br />
Day -1: “New iPad Pro sold out within hours of release.” <br />

### Stage 1: Supervised Fine-Tuning (SFT)

#### Goal and Method:

- Train an initial SFT model (M_E) to format outputs correctly and learn basic (i) label prediction of positive/ negative & (ii) explanation on the predicted label
- The explanation is given by a larger general-purpose LLM model (e.g. ChatGPT) and only the text data which the LLM predicts correctly will be used to train the initial SFT model 
- Note: The SFT model we obtain from this stage can output a basic form but is not yet optimized.  

#### Example:

Input data to LLM: the 5-day tweets given above (Xₜ)

LLM output (Yₜ = (label, explanation)):
    (i) Prediction: Positive
    (ii) Explanation: “Apple’s strong earnings and product demand outweigh short-term concerns about regulation and China sales.”

✅ Suppose this is correct (label is Positive), we save this (Xₜ, Yₜ) pair and use it to train an initial SFT model.

> 📌Remarks: The positive/ negative price prediction label is given by stock price movement in the price data collected

> 💬Comments: The SFT model (M_E) is like a student model supervised by the general purpose LLM model in this stage.

### Stage 2: Reward Model (r_θ) Training

#### Goal:
Prepare a reward model (r_θ) for reinforcement learning to optimize the SFT model in the next stage. 

#### Method:
- Given input Xₜ , The initial SFT model (M_E) from stage 1 is asked to generate predictions Yₜ = (label, explanation)) 
- If the predicted label is
    - **incorrect**: 
        there will be another general-purpose LLM model (e.g. ChatGPT) (M_R) comes in <br />
        -> M_R works as a 'reflector' which outputs verbal feedback (rₜ) explaining the mistakes <br />
        -> the feedback (rₜ) will be put together with Xₜ as the new input to the SFT model (M_E) in the next iteration to generate new improved predictions Yₜ <br />
       -> the iteration continues until correct predicted label is obtained
	- **correct**:
       all predictions Yₜ's obtained (both correct and incorrect) for Xₜ  will be used to train a reward model r_θ (which is also a neural network model) <br />
       -> The reward model r_θ will be trained to give <br />
           - higher score to correct predicted label <br />
           - lower score to incorrect predicted label <br />

> 💬Comments: Compared with direct supervised learning which only teach the target model to mimic existing answers, the reward model provides a flexible signal that allows fine-tuning for quality beyond just exact match (work as a proxy for human judgement) 

### Stage 3: Reinforcement learning to optimize the SFT model 

#### Goal:
Finetune the M_E model using reinforcement learning and reward policy r_θ through "Proximal Policy Optimization" (PPO). (The improved M_E model will be called π_RL)

#### Method:
Initialize policy π_RL with the weights of M_E <br />
-> Generate a response Ŷₜ using π_RL(Xₜ) <br />
-> Score it using Reward Model: r = r_θ(Xₜ, Ŷₜ) <br />
-> PPO adjusts the policy (π_RL) to increase the likelihood of high-reward outputs. <br />
-> Iterate over data until convergence. <br />

> 💬Comments: The improved M_E model (π_RL) is now reinforced to prefer outputs that resemble good self-reflected answers — better explanations and more accurate predictions.

## Future Improvements

- The sentimental prediction from SEP model can be used together as the price data in a time series for short-term stock price prediction (like the use in a previous [repo](https://github.com/ycyam-oliver/Transformer-Stock) of mine)

- Instead of the expensive ChatGPT model, other open-source LLM model on clouds can also be used as teacher model, which should makes it more economical in training

- Apart from making prediction on stock price, text data like financial reports and analysis report from major investment bank can be used in the input and train the model make prediction on other financial multiples like P/E, P/B etc. This could be more helpful in valuation of stock. 



