# DeepSentiment

## Introduction
Sentiment classification plays a crucial role in various applications such as social media analysis and customer feedback processing. In this study, we implement three approaches for sentiment classification: multi-channel convolutional neural networks (MC-CNN), graph convolutional networks (GCN), and adaptive multi-channel GCN (AM-GCN). These models have been proposed in separate papers and have shown promising results in capturing complex patterns in text data for sentiment analysis tasks.

## Usage

Clone the repo and install packages

```
git clone https://github.com/atcact/deep-sentiment.git
cd deep-sentiment
pip install -r requirements.txt
```

To run RNN and MC-CNN models:

```bash
python main.py --model [rnn/mccnn] --dataset [imdb/sts_gold]
```

To run GCN/AMGCN models:

```bash
cd models/amgcn
python train.py 
```

## Dataset

Graph we built for IMDB dataset:
https://drive.google.com/drive/folders/1RIXVnnVmBtm_vgGJYHaaD7AQsu8uNVz_?usp=drive_link


