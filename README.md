# What KAN I Spike: MambaOut

> "What can I say, Mamba out." — *Kobe Bryant, NBA farewell speech, 2016*

<p align="center">
<img src="https://raw.githubusercontent.com/yuweihao/misc/master/MambaOut/mamba_out.png" width="400"> <br>
<small>Image credit: https://www.ebay.ca/itm/264973452480</small>
</p>


This is the repo of the course project of 2024 fall brain-inspired intelligence.

This model is a modification of MambaOut model, modified form the implementation of MambaOut proposed by the paper "[MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2405.07992)". 

# Run 
## SMO, XSMO

1. See `val.sh`, modify the three arguments at top. Checkpoint file name should match model name.
2. `bash val.sh` and wait. You should see results in a minute.

## Explorations
```bash
python explorations/train.py <act_name>
```
act_name is either `gelu`, `srelu` or `seplu`. You should modify `DATASET_PATH` to fit your path.