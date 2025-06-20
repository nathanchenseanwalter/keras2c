============
Installation
============


keras2c can be downloaded from github: https://github.com/f0uriest/keras2c

On macOS, install the Xcode command line tools so ``make`` and ``clang`` are
available (run ``xcode-select --install``).  Windows users should install
``make`` and ``gcc`` through MSYS2/MinGW or Cygwin.

The Python requirements can be installed with pip:

.. code-block:: bash

    pip install -r requirements.txt

Alternatively, create a conda environment using the provided YAML file:

.. code-block:: bash

    conda env create -f environment.yml


Additional packages for building the documentation and running the tests are included in the conda environment, but can also be installed separately with pip:

.. code-block:: bash

    pip install -r docs/requirements.txt
    pip install -r tests/requirements.txt


By default, the tests compile code with ``gcc``.  Set the environment variable
``CC`` to override the compiler when running ``pytest`` or ``make``.  It is also
recommended to install ``astyle`` to automatically format the generated code.
