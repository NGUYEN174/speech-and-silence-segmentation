"""
Simple Statistics-based Speech/Silence Segmentation.

Reference:
    Simple Statistics method from the course/reference material.

STATUS: PLACEHOLDER
    This module is a placeholder waiting for the responsible student's
    implementation. It implements the BaseAlgorithm interface so the
    project can run without errors.

When the responsible student provides the implementation:
    1. Replace this file with the actual algorithm
    2. The class must inherit from BaseAlgorithm
    3. Implement train() and predict() methods
    4. No other project files need to be modified

See src/common/interfaces.py for the required interface.
"""

from src.common.interfaces import PlaceholderAlgorithm


class SimpleStatisticsAlgorithm(PlaceholderAlgorithm):
    """Placeholder for the Simple Statistics algorithm.
    
    This will be replaced by the responsible student.
    Currently returns no results and is skipped during evaluation.
    """
    
    def __init__(self):
        super().__init__(
            algorithm_name="Simple Statistics",
            algorithm_key="simple_statistics"
        )
