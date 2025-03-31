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

Prediction starts from annotation `a` and walks forward and backward with predicted frames `p(a, $)` and `p(a, ^)`


```
^PPPPPPPPPPPPPPPPPPPPPPAPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP$
                       ↑
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

`(^,a)` = `(a → ^)`:
```
^PPPPPPPPPPPA----------A---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

`(a,b)` = `(a → b) U (b → a)`:
```
^-----------FPPPPPPPPPPF---------------F-------------------$
            ↑          ↑               ↑
            a          b               c
```

`(b,c)` = `(b → c) U (c → b)`:
```
^-----------A----------APPPPPPPPPPPPPPPA-------------------$
            ↑          ↑               ↑
            a          b               c
```

`(c,$)` = `(c → $)`:
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
`(^,a)` = `C` # Clear of fluid (no fluid annotations between `^` and `a`):
```
^CCCCCCCCCCCC----------A---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

`(a,b)` = `(b → a)` # No fluid to track from `a` going forward:
```
^-----------CPPPPPPPPPPA---------------A-------------------$
            ↑          ↑               ↑
            a          b               c
```

`(b,c)` = `(b → c) U (c → b)`:
```
^-----------C----------APPPPPPPPPPPPPPPA-------------------$
            ↑          ↑               ↑
            a          b               c
```

`(c,$)` = `(c → $)`:
```
^-----------C----------A---------------APPPPPPPPPPPPPPPPPPP$
            ↑          ↑               ↑
            a          b               c
```