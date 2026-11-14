"""Pure analytical self-check; no pymoo required. Run before expensive runs."""
from di_barba_solenoid_benchmark import DiBarbaSolenoid
import numpy as np

model = DiBarbaSolenoid()
assert model.n_var == 10
assert model.K.shape == (41, 10)
assert np.allclose(model.evaluate(np.zeros(10)), (1.0, 0.0))
x = np.linspace(-0.5, 0.5, 10)
J = model.gradient(x)
for k in range(10):
    h = 1.e-6
    e = np.zeros(10)
    e[k] = h
    fd = (np.array(model.evaluate(x+e))-np.array(model.evaluate(x-e)))/(2*h)
    assert np.allclose(fd, J[:,k], rtol=2e-5, atol=1e-7), (k, fd, J[:,k])
print('OK: 10-D analytical model and both objective gradients')
