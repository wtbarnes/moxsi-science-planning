# reports

This directory contains a set of notebooks whose output should be preserved for future reference.
These should not be altered.
These are periodically converted to PDF format and added as releases to the GitHub repository.
To convert the notebooks to PDF,

```shell
$ jupyter-nbconvert --to=pdf --output-dir=pdfs/ *.ipynb
```