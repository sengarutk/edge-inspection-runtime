# Image Anomaly Map Spatial Aggregation Ablation

| ('aggregation_rule', '')   |   ('image_auroc', 'mean') |   ('image_auroc', 'std') |   ('image_ap', 'mean') |   ('image_ap', 'std') |
|:---------------------------|--------------------------:|-------------------------:|-----------------------:|----------------------:|
| gaussian_pooled_max        |                  0.858296 |                 0.230177 |               0.924124 |              0.120824 |
| global_max                 |                  0.858303 |                 0.229979 |               0.924331 |              0.120392 |
| percentile_95              |                  0.858202 |                 0.230203 |               0.924266 |              0.120515 |
| percentile_99              |                  0.85834  |                 0.22994  |               0.924302 |              0.120408 |
| top_1_percent_mean         |                  0.858329 |                 0.229968 |               0.92434  |              0.120384 |