---
language: en
datasets:
- laion2b
---

# OpenFlamingo-9B

[Blog post]() | [Code](https://github.com/mlfoundations/open_flamingo) | [Demo](https://7164d2142d11.ngrok.app)

OpenFlamingo is an open source implementation of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) models. 
OpenFlamingo-9B is built off of [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) and [LLaMA-7B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).


## Model Details
We freeze the pretrained vision encoder and language model, and then we train connecting Perceiver modules and cross-attention layers, following the original Flamingo paper. 

Our training data is a mixture of [LAION 2B](https://huggingface.co/datasets/laion/laion2B-en) and a large interleaved image-text dataset called Multimodal C4, which will be released soon.

The current model is an early checkpoint of an ongoing effort. This checkpoint has seen 5 million interleaved image-text examples from Multimodal C4 and 10 million samples from LAION 2B.

## Uses
OpenFlamingo-9B is intended to be used **for academic research purposes only.** Commercial use is prohibited, in line with LLaMA's non-commercial license.

### Bias, Risks, and Limitations
This model may generate inaccurate or offensive outputs, reflecting biases in its training data and pretrained priors. 

In an effort to mitigate current potential biases and harms, we have deployed a text content filter on model outputs in the OpenFlamingo demo. We continue to red-team the model to understand and improve its safety.

## Evaluation
We've evaluated this checkpoint on the validation sets for two vision-language tasks: COCO captioning and VQAv2. Results are displayed below.

**COCO (CIDEr)**

|0-shot|4-shot|8-shot|16-shot|32-shot|
|--|--|--|--|--|
|65.52|74.28|79.26|81.84|84.52|


**VQAv2 (VQA accuracy)**

|0-shot|4-shot|8-shot|16-shot|32-shot|
|---|---|---|---|---|
|43.55|44.05|47.5|48.87|50.34|
