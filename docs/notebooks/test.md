---
file_format: mystnb
kernelspec:
  name: python3
---

# Rendering Documentation with Jupytext + Myst-NB

## Math Test

Let's put some math here: $e = mc^2$, and also

$$
\begin{aligned}
x^2 + y^2 & = 1 \\
x & = \cos(t) \\
y & = \sin(t).
\end{aligned}
$$

Now let's try a macro and an equation reference:

$$
y_i = \xx_i'\bbe + \varepsilon_i.
$$ (eqn:lm)

If it works we should get {eq}`eqn:lm`.


(sec:code_test)=
## Code Test

```python
import numpy as np
x = np.array([1., 2., 3.])
x
```

## Cross-Referencing Test

A review of particle filters is provided in {cite:t}`doucet_johansen09`, but it does not discuss score and hessian calculations {cite:p}`poyiadjis_etal11`.  In addition, please see [code](sec:code_test).

## References

```{bibliography} biblio.bib
```
