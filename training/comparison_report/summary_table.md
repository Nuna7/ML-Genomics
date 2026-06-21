# Model comparison summary

| model       |   best_epoch |   best_val_loss |   val_mse |   val_distance_stratified_mse |   val_stratum_adjusted_corr |   total_epochs_trained |   total_train_time_min |
|:------------|-------------:|----------------:|----------:|------------------------------:|----------------------------:|-----------------------:|-----------------------:|
| CNN         |           36 |        0.183554 |  0.183554 |                      0.183554 |                    0.454729 |                     40 |                  504.9 |
| UNet        |           38 |        0.183999 |  0.183999 |                      0.183999 |                    0.457813 |                     40 |                  512   |
| Transformer |           37 |        0.19973  |  0.19973  |                      0.19973  |                    0.420093 |                     40 |                   32   |

**Note on fairness of this comparison**: best_epoch is chosen independently per model (lowest val_loss for that model's own training run), not a shared fixed epoch count, because the three architectures converge at different speeds (in particular, the Transformer's attention starts near-uniform and needs more optimization steps to sharpen -- see model_transformer.py). If one model was trained for far fewer total epochs than the others and its val_loss curve in comparison_curves.png has clearly not plateaued yet, its result here is not a fair final comparison -- train it further before concluding it is worse. Compare the val-loss-vs-wall-clock-time panel as well, since epoch count alone does not account for the fact that the Transformer and UNet have different per-epoch costs than the CNN.
