import numpy as np
import pytest

from hugin import petro


def test_gr_index_clipped():
    assert np.allclose(petro.gr_index(np.array([10, 30, 80, 200]), 30, 80), [0, 0, 1, 1])
    assert petro.gr_index(55, 30, 80) == pytest.approx(0.5)


def test_larionov_older_endpoints_and_below_linear():
    igr = np.linspace(0, 1, 11)
    v = petro.larionov_older(igr)
    assert v[0] == pytest.approx(0) and v[-1] == pytest.approx(0.99)
    assert np.all(v <= igr + 1e-12)


def test_density_porosity():
    assert petro.density_porosity(2.65) == pytest.approx(0)
    assert petro.density_porosity(1.0) == pytest.approx(1)
    assert petro.density_porosity(2.32, 2.65, 1.0) == pytest.approx(0.2)


def test_dn_rms():
    assert petro.density_neutron_porosity(0.2, 0.2) == pytest.approx(0.2)
    assert petro.density_neutron_porosity(0.3, 0.1) == pytest.approx(np.sqrt(0.05))


def test_effective_porosity_removes_shale():
    # pure shale -> zero effective porosity
    assert petro.effective_porosity(0.12, 0.35, 1.0, 0.12, 0.35) == pytest.approx(0)
    # clean sand -> unchanged
    assert petro.effective_porosity(0.2, 0.2, 0.0, 0.12, 0.35) == pytest.approx(0.2)


def test_archie_known_value():
    # Rw=0.05, phi=0.2, m=n=2, Rt=5 -> Sw = sqrt(0.05/(0.04*5)) = 0.5
    assert petro.archie_sw(5.0, 0.2, 0.05) == pytest.approx(0.5)
    assert petro.archie_sw(0.1, 0.2, 0.05) == 1.0  # clipped


@pytest.mark.parametrize("n", [1.8, 2.0, 2.4])
def test_simandoux_reduces_to_archie_when_clean(n):
    sw_s = petro.simandoux_sw(5.0, 0.2, 0.0, 0.05, 2.0, n=n)
    sw_a = petro.archie_sw(5.0, 0.2, 0.05, n=n)
    assert sw_s == pytest.approx(sw_a, abs=1e-6)


def test_simandoux_lower_than_archie_when_shaly():
    assert petro.simandoux_sw(5.0, 0.2, 0.3, 0.05, 2.0) < petro.archie_sw(5.0, 0.2, 0.05)


def test_simandoux_matches_closed_form_n2():
    rt, phi, vsh, rw, rsh = 8.0, 0.18, 0.25, 0.06, 2.5
    c = vsh / rsh
    closed = (rw / (2 * phi**2)) * (np.sqrt(c**2 + 4 * phi**2 / (rw * rt)) - c)
    assert petro.simandoux_sw(rt, phi, vsh, rw, rsh) == pytest.approx(closed, rel=1e-6)


def test_simandoux_broadcasts_over_draws():
    m = np.array([[1.8], [2.0], [2.2]])
    sw = petro.simandoux_sw(np.array([5.0, 10.0]), 0.2, 0.1, 0.05, 2.0, m=m)
    assert sw.shape == (3, 2)
    assert np.all(np.diff(sw[:, 0]) > 0)  # higher m -> higher Sw


def test_arps_and_rwa():
    assert petro.arps_rw(0.05, 100, 100) == pytest.approx(0.05)
    assert petro.arps_rw(0.05, 50, 100) < 0.05
    assert petro.rwa(1.0, 0.25) == pytest.approx(0.0625)


def test_flags_nan_fails():
    net, pay = petro.flags(np.array([0.1, np.nan, 0.1]), np.array([0.2, 0.2, 0.05]),
                           np.array([0.3, 0.3, 0.3]), 0.4, 0.1, 0.6)
    assert net.tolist() == [True, False, False] and pay.tolist() == [True, False, False]
