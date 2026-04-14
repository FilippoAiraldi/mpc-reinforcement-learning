from timeit import repeat

import numpy as np

rng = np.random.default_rng(0)
gradients = [rng.random(77) for _ in range(121)]
gradients = tuple(gradients)
glob = {"np": np, "gradients": gradients}

times = repeat("np.mean(gradients, 0)", repeat=100, number=1000, globals=glob)
print(np.mean(times), "+/-", np.std(times))

times = repeat("np.mean(np.stack(gradients), 0)", repeat=100, number=1000, globals=glob)
print(np.mean(times), "+/-", np.std(times))
