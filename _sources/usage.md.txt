# Usage

Here is a basic example illustrating how to use `lsnms`:

```python
import numpy as np
from lsnms import nms, wbc

# Create boxes: approx 30 pixels wide / high in Pascal VOC format:
# bbox = (x0, y0, x1, y1) with x1 >= x0, y1 >= y0
image_size = 10_000
n_predictions = 10_000
topleft = np.random.uniform(0.0, high=image_size, size=(n_predictions, 2))
wh = np.random.uniform(15, 45, size=topleft.shape)
boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
# Create scores
scores = np.random.uniform(0.0, 1.0, size=(len(boxes),))

# Create class_ids if multiclass, 3 classes here
class_ids = np.random.randint(0, 2, size=(len(boxes),))

# Apply NMS
# During the process, overlapping boxes are queried using a R-Tree, ensuring a log-time search
keep = nms(boxes, scores, iou_threshold=0.5)
# Or, in a multiclass setting
keep = nms(boxes, scores, iou_threshold=0.5, class_ids=class_ids)
boxes = boxes[keep]
scores = scores[keep]

# Apply WBC
pooled_boxes, pooled_scores, cluster_indices = wbc(boxes, scores, iou_threshold=0.5)
```
