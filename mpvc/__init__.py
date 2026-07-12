"""
MPVC - MediaPipe & OpenCV Detectors
Módulo profissional para detecção simplificada de mãos, rostos e malha facial.
"""

from .detector_maos import DetectorMaos
from.detector_rosto import DetectorRosto
from .malha_rosto import DetectorMalhaRosto

__all__ = [
    'DetectorMaos',
    'DetectorRosto',
    'DetectorMalhaRosto'
]

__version__ = '0.0.2'
__author__ = 'Mundo Python'
