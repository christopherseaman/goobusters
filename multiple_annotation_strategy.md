# Multiple Annotation Strategy

## North Star

We create a model capable of filling in predicted frames from a sparsely annotated scan.

Goal state:
- There is at least one annotation for each temporally/spatially contiguous pocket of free fluid
- There may be additional annotations for a given pocket or for frames clear of free fluid
- The model is able to identify situations where free fluid moves out of view and clear that portion of the mask
- Frames between two annotations are informed by predictions from both 

Known error states:
- Incomplete input annotations are assumed correct and will not be altered if incomplete

## Notation
- `^`: Beginning of scan/video (first frame to the right)
- `$`: End of scan/video (final frame to the left)
- `-`: Unannotated frame
- `F`: Fluid annotated frame (all fluid masked and verified)
- `C`: No fluid annotated frame (no fluid, verified)
- `P`: Predicted frames
- `a-z`: Annotation identifiers
- `(x,y)`: Set of predicted frames between annotation `x` and `y`
- `(x → y)`: Set of predicted frames between annotation `x` and `y` using annotation `x` to track fluid

## Single Annotation - Current State

```
^----------------------F-----------------------------------$
                       ↑
                       a
```

Prediction starts from annotation `a` and walks forward and backward with predicted frames `p(a, $)` and `p(a, ^)`


```
^PPPPPPPPPPPPPPPPPPPPPPFPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP$
                       ↑
                       a
```

## Multiple Annotation - MVP

### All Fluid Annotations

```
^-----------F----------F---------------F-------------------$
            ↑          ↑               ↑
            a          b               c
```

In this case, the frames are predicted as:
- `(^,a)` = `(a → ^)`
- `(a,b)` = `(a → b) U (b → a)`
- `(b,c)` = `(b → c) U (c → b)`
- `(c,$)` = `(c → $)`
