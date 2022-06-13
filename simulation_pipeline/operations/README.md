# Detector Pipeline

```python
class DetectorPipeline:
    ...
```

Whether the detector is for the pinhole or the dispersed image should depend on what is substituted for the wavelength response and projection modules.

## Wavelength Response Module

This module will convert spectral cube from physical units (e.g. photon or erg) to detector units

## Projection Module

Reproject the spectral cube into the detector frame