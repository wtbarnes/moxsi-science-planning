# Physics

```python
class PhysicsPipeline:

    def __init__(self, dem, spectra):
        self.dem = dem
        self. spectra = spectra

    def run(self):
        dem_cube = self.dem.run()
        spec_cube = self.spectra.run(dem_cube)
       return spec_cube
```

## DEM Module

The only requirement of the DEM module is that it produces a DEM cube.
Very generally, this would look like the following:

```python
class DemModule:

    def __init__(self, *args, **kwargs):
        ...

    def run(self):
        # Do some computation to return a  DEM cube of dimensions (nT, nX, nY)
        return dem_cube
```

In the case where we compute an inverted DEM from AIA and XRT, the subclass would look like the following

```python
class InvertedDem(DemModule):

    def __init__(self, date, dem_model_class):
        self.date = date
        self.dem_model_class = dem_model_class

    def fetch_data(self):
        # Use Fido to query the data
        return list_of_maps

    def build_collection(maps):
        # Reproject all maps to same coordinate frame
        # Place in a collection 
        return map_collection

    def get_responses(map_collection):
        # For each key in the map, get a wavelength response
        # and put it in a dictionary
        return response_dict

    def compute_dem(map_collection, response_dict):
        dem_model = self.dem_model_class(map_collection, response_dict)
        return dem_model.fit()

    def run(self):
        maps = self.fetch_data()
        map_collection = self.build_collection(maps)
        responses = self.get_responses(map_collection)
        dem_cube = self.compute_dem(map_collection, responses)
        return dem_cube
```

## Spectral Module
