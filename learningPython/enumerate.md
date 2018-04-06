A new built-in function, `enumerate()`, will make certain loops a bit clearer. `enumerate(thing)`, where thing is either an iterator or a sequence, returns a iterator that will return `(0, thing[0])`, `(1, thing[1])`, `(2, thing[2])`, and so forth.

A common idiom to change every element of a list looks like this:

```
for i in range(len(L)):
    item = L[i]
    # ... compute some result based on item ...
    L[i] = result

```

This can be rewritten using `enumerate()` as:

```
for i, item in enumerate(L):
    # ... compute some result based on item ...
    L[i] = result
```