<p align="center" width="100%">
<img src="https://ice.frostsky.com/2024/03/17/214a3af4a34a26be0a04e551e16b9364.webp"  width="40%" height="80%">
</p>
<div>

# Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models


<p align="left">
    <a href='https://github.com/liuzuyan' target='_blank'>Zuyan Liu<sup>*,1</sup></a>&emsp;
    <a href='https://github.com/dongyh20/' target='_blank'>Yuhao Dong<sup>*,1</sup></a>&emsp;
    <a href='https://raoyongming.github.io/' target='_blank'>Yongming Rao<sup>2,&#x2709</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=6a79aPwAAAAJ' target='_blank'>Jie Zhou<sup>1</sup></a>&emsp;
	<a href='https://scholar.google.com/citations?user=TN8uDQoAAAAJ' target='_blank'>Jiwen Lu<sup>1,&#x2709</sup></a>
</p>

<p align="left"><sup>1</sup>Tsinghua University &ensp; <sup>2</sup>Tencent&ensp; <sup>*</sup> Equal Contribution<sup>&ensp; &#x2709</sup>  Corresponding Author</p>

[Project Page](https://sites.google.com/view/chain-of-spot/) | [Arxiv Paper](https://arxiv.org/abs/???) | [Huggingface Model](https://huggingface.co/Zuyan/llava-CoS-13B)

## Chain-of-Spot

<p align="center" width="100%">
<img src="https://ice.frostsky.com/2024/03/19/7f7a1ebc8c38bcd1d48fb7737b352411.jpeg"  width="100%" height="100%">
</p>

**C**hain-**o**f-**S**pot **(CoS)** encourages Large Vision-Language Models to identify the **key region of interest** (ROI) in the image condition on the posed questions or instructions, and reasoning through an interactive manner.

This technique allows VLMs to access more detailed visual information without altering the original image resolution, thereby offering **multi-granularity** image features and improving the ability of visual understanding. 

## Updates

**[2024-03]**

1. 🤗 Introducing our project homepage: https://sites.google.com/view/chain-of-spot
2. 🤗 Check our [paper](https://arxiv.org/abs/???) introducing **Chain-of-Spot** in details. 
3. 🤗 Check our [model](https://huggingface.co/Zuyan/llava-CoS-13B) on huggingface.

## Get Started

1. **LLaVA Preparations**: We choose [LLaVA](https://github.com/haotian-liu/LLaVA) as our base model, so please follow the [instructions](https://github.com/haotian-liu/LLaVA/blob/main/README.md) to install essential codebase for running LLaVA. Or you can simply run the following scripts:

   ```
   git clone https://github.com/haotian-liu/LLaVA
   cd LLaVA
   pip install -e .
   ```

2. **Initial Weights**: We use [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [LLaVA-1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b) for finetuning, you may download these models and put them in the `./checkpoint` folder.

3. **Download Data**: The dataset structure is the same as used in LLaVA, and we provide json files to modify original LLaVA training dataset into our dataset in the following part. To correctly download the data, please check the [instructions](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning). 

   After downloading all of them, organize the data as follows in `./playground/data`

   ```
   ├── coco
   │   └── train2017
   ├── gqa
   │   └── images
   ├── ocr_vqa
   │   └── images
   ├── textvqa
   │   └── train_images
   └── vg
       ├── VG_100K
       └── VG_100K_2
   ```

4. **Training Data Preparations**: We migrate the brilliant work of [LRP++](https://github.com/hila-chefer/Transformer-MM-Explainability) to detect the correct ROI corresponding to a single question or instruction. You can directly download our generated dataset to reproduce our results from [Google Drive](https://drive.google.com/file/d/1QT2krgjSBNxm-qMez3ytFAeEvq7rMfNO/view?usp=sharing). You may also follow the [Notebook](https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_Explainability.ipynb) to prepare your own data.

5. **Evaluations on Various Benchmarks**: We follow the [Evaluation Docs](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) in LLaVA to conduct our experiments. If you find it laborious and complex, please check [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for faster evaluation.

6. **Start Training!** The finetuning process takes around 20 hours on 8*A100 (80G) for LLaVA-1.5-13B. We fintune LLaVA-1.5 using Deepspeed Zero-3, you can directly run the [scripts](https://github.com/dongyh20/Chain-of-Spot/tree/master/scripts/v1_5/finetune_CoS_13b.sh) to launch training:
```bash ./scripts/v1_5/finetune_CoS_13b.sh```

**Contact: Leave issue or contact `liuzuyan19@gmail.com` and `dongyh20@mails.tsinghua.edu.cn`. We are on call to respond.**

## Quantitative and Qualitative Results

### Comparisons with State-of-the-Art Models

Our **Chain-of-Spot** (CoS)  consistently improves the vanilla LLaVA-1.5 in all the benchmarks under different language model sizes. The best results are highlighted **bold**.

| Method | Language | VQA-v2 |       GQA        |    VizWiz     | SQA | Text-VQA |      OKVQA       |
| :---------------------- | :--------- | :------------------: | :----------------: | :-------------: | :----------------: | :----------------: | :----------------: |
| LLaVA-1.5-7B            | Vicuna-7B  |         78.5         |        62.0        |      50.0       |        66.8        |        58.2        |        57.9        |
| LLaVA-1.5-7B + **CoS**  | Vicuna-7B  |         80.7         |        63.7        |      50.8       |        68.2        |        60.9        |        58.4        |
| LLaVA-1.5-13B           | Vicuna-13B |         80.0         |        63.3        |      53.6       |        71.6        |        61.3        |        60.9        |
| LLaVA-1.5-13B + **CoS** | Vicuna-13B |  **81.8**  | **64.8** | **58.0** | **71.9** |  **62.4**   | **62.9** |

LLaVA-1.5 with **Chain-of-Spot** (CoS) a  achieves state-of-the-art performance on all the multimodal benchmarks, surpassing LVLMs by a large margin. The best results are highlighted **bold**.

| Method                | Language |       SEED       |      SEED_Img      |         MME          |       MMB        |       POPE       |      MM-Vet      |
| :---------------------- | :--------- | :----------------: | :----------------: | :--------------------: | :----------------: | :----------------: | :----------------: |
| LLaVA-1.5-7B            | Vicuna-7B  |        58.6        |        66.1        |         1510.7         |        64.3        |        85.9        |        30.5        |
| LLaVA-1.5-7B + **CoS**  | Vicuna-7B  |        59.7        |        67.1        |         1501.1         |        64.4        | **86.4** |        30.8        |
| LLaVA-1.5-13B           | Vicuna-13B |        61.6        |        68.2        |         1531.3         |        67.7        |        85.9        |        35.4        |
| LLaVA-1.5-13B + **CoS** | Vicuna-13B | **62.3** | **69.6** | **1546.1** | **68.2** |        86.1        | **37.6** |

### Visualizations

<p align="center" width="100%">
<img src="https://ice.frostsky.com/2024/03/19/adcb4e56756d0d3ad0e011fcb9a71090.jpeg"  width="100%" height="100%">
</p>

**Visualizations on Chain-of-Spot.** Chain-of-Spot shows the reasonable region of interest condition on the given questions.

<p align="center" width="100%">
<img src="https://ice.frostsky.com/2024/03/19/7e1d9aa58cb178f5a4a7d8edb479f95d.jpeg"  width="100%" height="100%">
</p>

**Generation comparisons after implementing Chain-of-Spot.** Chain-of-Spot corrects the focus and the answers of LLaVA model on complex visual question cases.

## Citation

If you found this repository useful, please consider citing:

``` 
Arxiv paper link (coming soon)
```

## Acknowledgements

We thank the [LLaVA](https://github.com/haotian-liu/LLaVA) team for their great contribution to the open-source VLM community.
