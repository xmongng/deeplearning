# RealWaste — Data audit

Archive SHA-256: `1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3`

Images: 4752; readable and retained: 4752; excluded: 0.
Candidate duplicate pairs: 14; cross-label groups: 4.

| Class | Before | After |
|---|---:|---:|
| Cardboard | 461 | 461 |
| Food Organics | 411 | 411 |
| Glass | 420 | 420 |
| Metal | 790 | 790 |
| Miscellaneous Trash | 495 | 495 |
| Paper | 500 | 500 |
| Plastic | 921 | 921 |
| Textile Trash | 318 | 318 |
| Vegetation | 436 | 436 |

Exact and near-duplicate images are retained but grouped in the same split.
pHash candidates are conservative leakage groups, not verified duplicate labels.
Review data/review.html and suspicious labels before freezing splits.
Record exclusions/relabels in data/review_decisions.csv and rerun audit.
Automatic checks cannot prove that all labels are correct or all near duplicates were found.
