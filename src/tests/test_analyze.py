"""Tests for analysis and plotting functionality."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCriticalEpsilon:
    """Test critical epsilon calculation."""

    def test_find_critical_epsilon_basic(self):
        """Test finding critical epsilon when rate drops below 50%."""
        from analyze_results import find_critical_epsilon

        epsilon_values = [0.01, 0.02, 0.03, 0.05, 0.10]
        verified_rates = [95.0, 80.0, 60.0, 40.0, 20.0]

        critical = find_critical_epsilon(epsilon_values, verified_rates, threshold=50.0)

        # Should be between 0.03 (60%) and 0.05 (40%)
        assert critical is not None
        assert 0.03 < critical < 0.05

    def test_find_critical_epsilon_never_drops(self):
        """Test when rate never drops below threshold."""
        from analyze_results import find_critical_epsilon

        epsilon_values = [0.01, 0.02, 0.03]
        verified_rates = [95.0, 85.0, 75.0]

        critical = find_critical_epsilon(epsilon_values, verified_rates, threshold=50.0)
        assert critical is None

    def test_find_critical_epsilon_drops_immediately(self):
        """Test when rate drops below threshold at first epsilon."""
        from analyze_results import find_critical_epsilon

        epsilon_values = [0.01, 0.02, 0.03]
        verified_rates = [40.0, 30.0, 20.0]

        critical = find_critical_epsilon(epsilon_values, verified_rates, threshold=50.0)
        assert critical == 0.01


class TestDataExtraction:
    """Test verification data extraction."""

    def test_extract_verification_data(self):
        """Test extracting data from results dict."""
        from analyze_results import extract_verification_data

        results = {
            "metadata": {},
            "results": {
                "robustness": [
                    {"epsilon": 0.01, "verified_rate": 90.0},
                    {"epsilon": 0.02, "verified_rate": 80.0},
                ],
                "safety": [
                    {"epsilon": 0.01, "verified_rate": 95.0},
                    {"epsilon": 0.02, "verified_rate": 85.0},
                ],
            },
        }

        robustness, safety = extract_verification_data(results)

        assert robustness[0.01] == 90.0
        assert robustness[0.02] == 80.0
        assert safety[0.01] == 95.0
        assert safety[0.02] == 85.0

    def test_extract_empty_results(self):
        """Test extracting from empty results."""
        from analyze_results import extract_verification_data

        results = {"metadata": {}, "results": {}}
        robustness, safety = extract_verification_data(results)

        assert robustness == {}
        assert safety == {}


class TestLoadResults:
    """Test loading verification results."""

    def test_load_verification_results(self):
        """Test loading results from JSON file."""
        from analyze_results import load_verification_results

        test_results = {
            "metadata": {"epsilon_values": [0.01, 0.02]},
            "results": {
                "robustness": [{"epsilon": 0.01, "verified_rate": 90.0}],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_results, f)
            temp_path = Path(f.name)

        try:
            loaded = load_verification_results(temp_path)
            assert loaded["metadata"]["epsilon_values"] == [0.01, 0.02]
            assert len(loaded["results"]["robustness"]) == 1
        finally:
            temp_path.unlink()


class TestPlotGeneration:
    """Test plot generation (without actually displaying)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample verification data for plotting."""
        robustness = {0.01: 95.0, 0.02: 85.0, 0.03: 70.0, 0.05: 45.0, 0.10: 20.0}
        safety = {0.01: 98.0, 0.02: 92.0, 0.03: 80.0, 0.05: 55.0, 0.10: 30.0}
        return robustness, safety

    def test_plot_verified_vs_epsilon(self, sample_data):
        """Test generating verification rate plot."""
        from analyze_results import plot_verified_vs_epsilon

        robustness, safety = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            plot_verified_vs_epsilon(robustness, safety, output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_comparison_with_nsga3(self, sample_data):
        """Test generating NSGA-III comparison plot."""
        from analyze_results import plot_comparison_with_nsga3

        robustness, safety = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_comparison.png"
            plot_comparison_with_nsga3(robustness, safety, output_path)
            assert output_path.exists()

    def test_plot_certified_accuracy(self, sample_data):
        """Test generating certified accuracy plot."""
        from analyze_results import plot_certified_accuracy

        robustness, _ = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_certified.png"
            plot_certified_accuracy(robustness, base_accuracy=99.97, output_path=output_path)
            assert output_path.exists()


class TestReportGeneration:
    """Test summary report generation."""

    def test_generate_summary_report(self):
        """Test generating text summary report."""
        from analyze_results import generate_summary_report

        robustness = {0.01: 95.0, 0.02: 85.0, 0.03: 70.0, 0.05: 45.0}
        safety = {0.01: 98.0, 0.02: 90.0, 0.03: 75.0, 0.05: 50.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generate_summary_report(robustness, safety, output_path, base_accuracy=99.97)

            assert output_path.exists()

            content = output_path.read_text()
            assert "VERIFICATION ANALYSIS REPORT" in content
            assert "LOCAL ROBUSTNESS" in content
            assert "SAFETY PROPERTY" in content
            assert "NSGA-III" in content
