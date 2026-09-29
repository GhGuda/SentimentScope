# SentimentScope

A transformer encoder trained from scratch to classify IMDB movie reviews as positive or negative.

No pretrained weights are used. The model is randomly initialised and learns entirely from the 25,000 labelled training reviews. Only the tokenizer (`bert-base-uncased`) is borrowed from a pretrained source.

**Final test accuracy: 82.41%** on the held-out 25,000-review test set, against a 50.52% untrained baseline.

## Contents

| File | Description |
|---|---|
| `SentimentScope.ipynb` | The full notebook: data loading, EDA, model, training, evaluation, inference |
| `sentimentscope_model.pt` | Trained checkpoint (weights plus the config needed to rebuild the model) |

## Model

| | |
|---|---|
| Architecture | 4 transformer blocks, 4 attention heads, `d_embed` 128 |
| Parameters | ~4.7M, of which 3.9M is the token embedding table |
| Context | 256 sub-word tokens |
| Pooling | Masked mean over non-padding positions |
| Head | Linear 128 to 2, no bias |
| Training | 12 epochs, AdamW, one-cycle LR, dropout 0.2, weight decay 0.1, label smoothing 0.05 |

Requires an NVIDIA GPU for training. A full run takes roughly two hours on a Tesla T4. Inference runs fine on CPU.

## Running it

```bash
pip install torch transformers pandas matplotlib numpy
```

The notebook expects `aclImdb_v1.tar.gz` in the same directory and extracts it in cell 3. Download it from [Stanford AI](https://ai.stanford.edu/~amaas/data/sentiment/) if you do not already have it.

Then run the notebook top to bottom.

To skip training and use the saved checkpoint instead, run only the cells that define the tokenizer, `MAX_LENGTH`, and the model classes, then:

```python
predictor = SentimentPredictor("sentimentscope_model.pt", tokenizer)
predictor.predict(["An absolute masterpiece.", "A complete waste of time."])
# [{'label': 'positive', 'confidence': 0.977},
#  {'label': 'negative', 'confidence': 0.979}]
```

The checkpoint stores its own config, so the architecture is reconstructed without repeating any hyperparameters.

## What moved the number

Three runs, each change made in response to something measured rather than guessed.

| Run | Change | Test accuracy |
|---|---|---|
| A | Baseline: causal attention, unmasked mean pooling, 3 epochs | 76.78% |
| B | Bidirectional attention, masked mean pooling, 10 epochs | 78.50% |
| C | 256-token context, dropout 0.2, weight decay 0.1, label smoothing, one-cycle LR, 12 epochs | **82.41%** |

**Run A to B (+1.72).** The starter code applies a causal mask, so each token can only attend to tokens before it. That is required for text generation, where seeing the next token would be cheating, but it discards information in a classifier: the whole review is available at inference time. Removing the mask and excluding `[PAD]` positions from the pooled representation recovered that.

**Run B to C (+3.91).** The length histogram in the EDA showed a median review of about 174 words against a 128-token budget. Sub-word tokenization emits roughly 1.3 tokens per word, so 128 tokens covered only about 95 words and truncated most reviews before their closing verdict. Doubling the context to 256 was the single largest gain. Epoch 1 alone scored 77.48% versus 71.04% in run A.

Run B also overfit badly, peaking at epoch 6 and declining for four epochs after. The regularization added in run C removed that collapse: validation reached 85.60% at epoch 7 and held near 85% through epoch 12.

## The ceiling

Training loss in run C settled around 0.15, close to the 0.117 floor imposed by label smoothing, while validation stayed in the mid-80s. The model fits the training set almost perfectly and generalises no further.

That is not a tuning problem. It is what 22,500 examples can teach a randomly initialised network. Fine-tuned BERT reaches about 93% on this same dataset, and the difference is pretraining on billions of words rather than anything architectural.

## Data

[Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011). 25,000 labelled training reviews and 25,000 test reviews, balanced across both classes. The training set is split 90/10 into 22,500 training and 2,500 validation reviews.
