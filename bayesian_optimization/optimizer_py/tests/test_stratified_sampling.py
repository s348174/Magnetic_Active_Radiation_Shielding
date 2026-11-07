"""Sampling mathematics and the Python/C++ stratified estimator contract."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from LogDataSampler import build_sampler, sample_all_particles_stratified
from input import FLUX_DATA, ENERGY_BINS
import simulator


class SamplingTests(unittest.TestCase):
    def test_exact_linear_density(self):
        # f(E)=E on [1,3] has mean 13/6, not the midpoint of its inverse knots.
        sampler = build_sampler(np.array([1., 3.]), np.array([1., 3.]),
                                np.random.default_rng(67))
        samples = sampler(200000)
        self.assertLess(abs(samples.mean() - 13/6), .004)

    def test_zero_density_gap(self):
        sampler = build_sampler(np.array([1.,2.,3.,4.]), np.array([1.,0.,0.,1.]),
                                np.random.default_rng(67))
        samples = sampler(10000)
        self.assertFalse(np.any((samples > 2) & (samples < 3)))

    def test_log_bins_and_nonuniform_counts(self):
        df = pd.DataFrame({'Energy': [1.,100.], 'proton': [1.,100.]})
        samples, strata = sample_all_particles_stratified(df, 11, 2, np.random.default_rng(67))
        np.testing.assert_allclose(strata['proton'], [[6,49.5],[5,4950.]])
        self.assertTrue(np.all((samples['proton'][:6] >= 1) & (samples['proton'][:6] <= 10)))
        self.assertTrue(np.all((samples['proton'][6:] >= 10) & (samples['proton'][6:] <= 100)))

    def test_weighted_integral(self):
        # Integral E*f(E) dE for f(E)=E on [1,100] is (100^3-1)/3.
        df = pd.DataFrame({'Energy': [1.,100.], 'proton': [1.,100.]})
        samples, strata = sample_all_particles_stratified(df, 40000, 2, np.random.default_rng(67))
        offset, estimate = 0, 0.
        for count, flux in strata['proton']:
            count = int(count)
            estimate += flux * samples['proton'][offset:offset+count].mean()
            offset += count
        exact = (100**3 - 1)/3
        self.assertLess(abs(estimate/exact - 1), .005)

    def test_empty_bins(self):
        df = pd.DataFrame({'Energy': [1.,10.,100.], 'proton': [1.,0.,0.]})
        samples, strata = sample_all_particles_stratified(df, 7, 2, np.random.default_rng(67))
        np.testing.assert_allclose(strata['proton'], [[7,4.5]])
        self.assertTrue(np.all(samples['proton'] <= 10))

    def test_invalid_budget_and_log_domain(self):
        df = pd.DataFrame({'Energy': [1.,100.], 'proton': [1.,1.]})
        with self.assertRaises(ValueError):
            sample_all_particles_stratified(df, 3, 2)
        df['Energy'] = [0.,100.]
        with self.assertRaises(ValueError):
            sample_all_particles_stratified(df, 10, 2)

    def test_original_spectra(self):
        samples, strata = sample_all_particles_stratified(FLUX_DATA, 1000, ENERGY_BINS,
                                                         np.random.default_rng(67))
        self.assertEqual(len(samples), 28)
        for species in samples:
            self.assertEqual(len(samples[species]), 1000)
            self.assertEqual(int(strata[species][:,0].sum()), 1000)
            self.assertTrue(np.all(strata[species][:,0] >= 2))
            expected = np.trapezoid(FLUX_DATA[species], FLUX_DATA['Energy'])
            np.testing.assert_allclose(strata[species][:,1].sum(), expected, rtol=1e-12)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.old_dir = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory(prefix='mars-strata-')
        root = Path(self.tmp.name)
        (root/'data').mkdir()
        (root/'optimizer').mkdir()
        self.particles = root/'data'/'particles_input.csv'
        self.particles.write_text('i,particleName,m,q\n1,proton,1,1\n')
        (root/'data'/'flux_data_clean.csv').write_text('Energy,proton\n1,1\n1000,1\n')
        os.chdir(root/'optimizer')
        self.centers = np.array([[3.,0.,0.]])
        self.normals = np.array([[0.,0.,1.]])
        self.samples = {'proton': np.array([2.,4.,20.,40.,200.,400.])}
        self.strata = {'proton': np.array([[2.,9.],[2.,90.],[2.,900.]])}

    def tearDown(self):
        os.chdir(self.old_dir)
        self.tmp.cleanup()

    def run_simulation(self, samples, strata=None):
        args = (67, len(samples['proton']), 1, 0., .25, self.centers, self.normals, samples)
        if strata is not None:
            args += (strata,)
        return np.array(simulator.launch_simulation(*args))

    def test_stratified_mean_and_variance(self):
        # Legacy calls provide independently computed bin moments, weighted by total flux 999.
        expected = np.zeros(2)
        for index, (count, flux) in enumerate(self.strata['proton']):
            block = self.samples['proton'][2*index:2*index+int(count)]
            mean, variance = self.run_simulation({'proton': block})
            expected += [mean * flux/999, variance * (flux/999)**2]
        result = self.run_simulation(self.samples, self.strata)
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        self.assertGreater(result[1], 0)

    def test_multiple_species(self):
        (Path('../data')/'flux_data_clean.csv').write_text('Energy,proton,Fe56\n1,1,2\n1000,1,2\n')
        one = self.run_simulation(self.samples, self.strata)
        self.particles.write_text('i,particleName,m,q\n1,proton,1,1\n2,Fe56,56,26\n')
        samples = dict(self.samples, Fe56=self.samples['proton'].copy())
        iron_bins = self.strata['proton'].copy()
        iron_bins[:,1] *= 2
        strata = dict(self.strata, Fe56=iron_bins)
        result = self.run_simulation(samples, strata)
        np.testing.assert_allclose(result, one * [1+2*56, 1+(2*56)**2], rtol=1e-10)

    def test_invalid_metadata(self):
        invalid = [np.array([[1.,999.]]), np.array([[2.5,999.]]),
                   np.array([[2.,999.]]), np.array([[6.,-999.]]),
                   np.array([[6.,1.]]), np.array([6.,999.])]
        for bins in invalid:
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                self.run_simulation(self.samples, {'proton': bins})

    def test_missing_species_is_rejected(self):
        self.particles.write_text('i,particleName,m,q\n1,proton,1,1\n2,Fe56,56,26\n')
        with self.assertRaises(ValueError):
            self.run_simulation(self.samples, self.strata)


if __name__ == '__main__':
    unittest.main(verbosity=2)
