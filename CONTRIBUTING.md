# Contributing to NMF Sound Localizer

Thank you for your interest in contributing to NMF Sound Localizer! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/speechlab/nmf-sound-localizer/issues) to avoid duplicates.

**When submitting a bug report, please include:**
- Clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version, package versions)
- Minimal code example demonstrating the issue
- Error messages or stack traces

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check existing issues and discussions first
- Clearly describe the enhancement and its motivation
- Provide examples of how it would be used
- Consider the scope and complexity of the change

### Pull Requests

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass
6. **Update** documentation if needed
7. **Submit** a pull request

## 🛠️ Development Setup

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/nmf-sound-localizer.git
cd nmf-sound-localizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=nmf_localizer --cov-report=html

# Run specific test file
pytest tests/test_localizer.py -v
```

### Code Quality

```bash
# Format code
black nmf_localizer tests examples

# Sort imports
isort nmf_localizer tests examples

# Lint code
flake8 nmf_localizer tests examples

# Type checking
mypy nmf_localizer --ignore-missing-imports
```

### Pre-commit Hooks

We recommend using pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📝 Code Standards

### Python Code Style
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Type hints are required for public APIs

### Documentation
- Docstrings for all public functions and classes
- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Include parameter types and return values
- Add examples for complex functions

### Testing
- Write tests for new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Include both unit and integration tests

## 🏗️ Project Structure

```
nmf_localizer/
├── __init__.py              # Package initialization
├── config/                  # Configuration management
│   ├── __init__.py
│   └── defaults.py
├── core/                    # Core algorithms
│   ├── __init__.py
│   ├── data_processor.py    # Data processing
│   ├── usm_trainer.py       # USM training
│   ├── localizer.py         # NMF localization
│   └── evaluator.py         # Performance evaluation
├── pipeline/                # High-level interfaces
│   ├── __init__.py
│   ├── full_pipeline.py     # Complete pipeline
│   └── experiment_runner.py # Batch experiments
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── audio_utils.py       # Audio processing
│   ├── math_utils.py        # Mathematical utilities
│   └── visualization.py     # Plotting functions
└── io/                      # Input/output
    ├── __init__.py
    ├── loaders.py           # Data loading
    └── savers.py            # Data saving
```

## 🧪 Testing Guidelines

### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical operations

### Test Data
- Use small synthetic datasets for unit tests
- Mock external dependencies where appropriate
- Include edge cases and error conditions
- Test both CPU and GPU code paths (when available)

### Example Test Structure

```python
import pytest
import torch
from nmf_localizer import NMFSoundLocalizer, NMFConfig

class TestNMFSoundLocalizer:
    def test_initialization(self):
        """Test localizer initialization."""
        config = NMFConfig(beta=0.0)
        localizer = NMFSoundLocalizer(config)
        assert localizer.config.beta == 0.0
    
    def test_localize_single_source(self):
        """Test single source localization."""
        # Test implementation...
        pass
        
    @pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
    def test_different_beta_values(self, beta):
        """Test localization with different beta values."""
        # Test implementation...
        pass
```

## 📚 Documentation

### API Documentation
- All public classes and functions must have docstrings
- Use clear, descriptive names
- Include type hints
- Provide usage examples

### README and Guides
- Keep README.md up to date
- Add examples for new features
- Update installation instructions if needed
- Include performance considerations

## 🚀 Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist
- [ ] All tests pass on all supported Python versions
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version number is bumped in setup.py
- [ ] Create GitHub release with changelog
- [ ] Package is published to PyPI (automated via CI)

## 🎯 Focus Areas

We're particularly interested in contributions in these areas:

### High Priority
- **Performance Optimization**: GPU acceleration, memory efficiency
- **Algorithm Improvements**: New NMF variants, better convergence
- **Evaluation Metrics**: Additional performance measures
- **Documentation**: Tutorials, examples, API documentation

### Medium Priority  
- **Visualization**: Interactive plots, better graphics
- **Data Processing**: New audio formats, preprocessing options
- **Configuration**: YAML/JSON config files, parameter validation
- **Command Line Interface**: CLI for common operations

### Future Ideas
- **Neural Networks**: Integration with deep learning models
- **Real-time Processing**: Streaming audio support
- **Multi-modal**: Integration with visual localization
- **Benchmarking**: Standard evaluation datasets

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/speechlab/nmf-sound-localizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/speechlab/nmf-sound-localizer/discussions)
- **Email**: For academic collaboration: contact@speechlab.example

## 📜 Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Please read and follow it to ensure a welcoming environment for all contributors.

## 🙏 Recognition

Contributors will be:
- Listed in the project README
- Included in release notes
- Acknowledged in academic publications (for significant contributions)
- Invited to co-author papers (for major contributions)

Thank you for contributing to NMF Sound Localizer! 🎉