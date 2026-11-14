import numpy as np
from di_barba_solenoid_benchmark import DiBarbaSolenoid

def test_field_objectives_and_gradients():
    m = DiBarbaSolenoid()
    assert m.n_var == 10
    assert np.allclose(m.evaluate(np.zeros(10)),(1.,0.))
    x=np.linspace(-.3,.3,10)
    for i in (0,5,9):
        e=np.zeros(10);e[i]=1e-6
        finite=(np.array(m.evaluate(x+e))-np.array(m.evaluate(x-e)))/(2e-6)
        np.testing.assert_allclose(finite, m.gradient(x)[:,i],rtol=1e-5,atol=1e-7)
