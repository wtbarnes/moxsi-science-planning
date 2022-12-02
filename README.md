# moxsi-science-planning

Notebooks and materials related to MOXSI, including the modeling and coalignment pipelines.

## Settuing Up on HelioCloud

1. Start an instance on https://dashboard.hsdcloud.org/
2. SSH into running instance
   ```
   $ ssh -i ".nasa-laptop.pem" -L 9998:localhost:8888 ubuntu@ec2-44-204-84-38.compute-1.amazonaws.com
   ```
3. Install mamba https://github.com/conda-forge/miniforge#install
4. Set up environment

## Starting Up Jupyter lab

```
$ jupyter lab --port=8888 --no-browser --ip=*
```