# Multiple Annotation Strategy

**In-practice, while creating densely-annotated dataset:**

1. Model predicts from current annotations set
2. Predictions pushed to MD.ai
  A. Replacing previous predictions or at least able to retrieve only latest predictions
  B. Predicted frames clearly differentiable from human-validated ones
4. Correction annotations added by subject matter expert
5. Go to 1.

**Note:**
- Only need clear-of-fluid `C` and fluid `F` annotations, fluid annotations must be complete (all fluid masked)
- If no annotations between `C` and end of scan, then nothing to track
- Need to combine two sets of predictions between `F` annotations
- Only one set of predictions between `F` and `C` annotations based on the fluid in `F`

## North Star

We create a model capable of filling in predicted frames from a sparsely annotated scan.

Goal state:
- There is at least one annotation for each temporally/spatially contiguous pocket of free fluid
- There may be additional annotations for a given pocket or for frames clear of free fluid
- The model is able to identify situations where free fluid moves out of view and clears that portion of the mask
- The model is able to track across apparent "splits" and "joins" of free fluid pockets
- Frames between two fluid annotations are informed by predictions from both
  - There _may_ be annotations for frames clear of fluid
  - Frames between fluid and clear annotations are only informed by the fluid annotation (no fluid to track from clear frame)
  - If no fluid annotated between clear and first/final frame, then those frames are also clear (no fluid to track from clear frame)
- Combining two predictions from two fluid annotations for the same predicted frame should be cheap and simple: e.g., union, weighted sum or average
  - Note: This relies of the predictions to be of high quality

Known error states:
- Incomplete input annotations are assumed correct and will not be altered if incomplete

## Notation
- `^`: Beginning of scan/video (first frame to the right)
- `$`: End of scan/video (final frame to the left)
- `-`: Unannotated frame
- `A`: Fluid annotated frame (all fluid masked and verified)
- `C`: No fluid annotated frame (no fluid, verified)
- `P`: Predicted frames
- `a-z`: Annotation identifiers
- `(x,y)`: Set of predicted frames between annotation `x` and `y`
- `(x → y)`: Set of predicted frames between annotation `x` and `y` using annotation `x` to track fluid
- `(x → y) U (y → x)`: Combination (possibly union) of two sets of predicted frames

## Single Annotation - Current State

```
^----------------------F-----------------------------------$
                       ↑
                       a
```

Prediction starts from annotation `a` and walks forward and backward with predicted frames

#### `(a → $)` and `(a → ^)`
```
^PPPPPPPPPPPPPPPPPPPPPPAPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP$
      (a → ^)          ↑             (a → $)
                       a
```

## Multiple Annotation - MVP


### All Fluid Annotations

```
^-----------A----------A---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

In this case, the frames are predicted as:

#### `(^,a)` = `(a → ^)`
Predict from a single annotation to the initial frame
```
^PPPPPPPPPPPA----------A---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

#### `(a,b)` = `(a → b) U (b → a)`
Combine predictions from two annotated frames, walking towards each other
```
^-----------FPPPPPPPPPPF---------------F-------------------$
            ↑          ↑               ↑
            a          b               c
```

#### `(b,c)` = `(b → c) U (c → b)`
Combine predictions from two annotated frames, walking towards each other
```
^-----------A----------APPPPPPPPPPPPPPPA-------------------$
            ↑          ↑               ↑
            a          b               c
```

#### `(c,$)` = `(c → $)`
Predict from a single annotation to the final frame
```
^-----------F----------F---------------APPPPPPPPPPPPPPPPPPP$
            ↑          ↑               ↑
            a          b               c
```

### Mixed Fluid/Clear Annotations

```
^-----------C----------A---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```


In this case, the frames are predicted as:

#### `(^,a)` = `C`
Clear of fluid (no fluid annotations between `^` and `a`)
```
^CCCCCCCCCCCC----------A---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

#### `(a,b)` = `(b → a)`
Track only from `b` walking towards `a` because `a` has no fluid to track going forward
```
^-----------CPPPPPPPPPPA---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

#### `(b,c)` = `(b → c) U (c → b)`
Track as usual between annotations
```
^-----------C----------APPPPPPPPPPPPPPPA-------------------$
            ↑          ↑               ↑
            a          b               c
```

#### `(c,$)` = `(c → $)`
Track as usual from a single annotation
```
^-----------C----------A---------------APPPPPPPPPPPPPPPPPPP$
            ↑          ↑               ↑
            a          b               c
```
