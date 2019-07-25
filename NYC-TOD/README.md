# NCY-TOD dataset

NCY-TOD is a large scale benchmark proposed in **[Contextualized Spatial-Temporal Network for Taxi Origin-Destination Demand Prediction](https://ieeexplore.ieee.org/abstract/document/8720246)**. To the best of our knowledge, it is the first benchmark of taxi origin-destination demand prediction. You can download this dataset from [dropbox](https://www.dropbox.com/s/ft4i0i0bysoox55/NYC-TOD.tar.gz?dl=0/)

## Region Partition 
We can get the geographic coordinates of all regions by running 
```python grid_split.py 
```

The generated ```mymap.html``` records many ```PolylineCoordinates```, each of which is the longitude and latitude of a region.

```
var PolylineCoordinates = [
new google.maps.LatLng(40.737886, -73.973962),
new google.maps.LatLng(40.745005, -73.968711),
new google.maps.LatLng(40.741141, -73.959645),
new google.maps.LatLng(40.734022, -73.964896),
];
```

## OD Demand Data
The OD demand of every time interval is recorded in `oddata.npy`.

## Meteorological Data
`weather.npy` records the temperature, windchill, humidity, visibility, wind speed, precipitation and 23 types of weather conditions of NYC during every time interval.

## Meta Data
The meta data is not used in this work, but we can explore its effect in future works.


If you use this benchmark for your research, please cite our work:

```
@article{liu2019contextualized,
  title={Contextualized Spatial-Temporal Network for Taxi Origin-Destination Demand Prediction},
  author={Liu, Lingbo and Qiu, Zhilin and Li, Guanbin and Wang, Qing and Ouyang, Wanli and Lin, Liang},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
