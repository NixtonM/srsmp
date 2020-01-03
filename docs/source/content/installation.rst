Installation Process
====================
The following section gives step-by-step instructions to set up **SRSMP** under a *Windows environment*.


Prerequisites
-------------
As mentioned before, **SRSMP** is currently only supported within *Windows*.
The following prerequisits must be fulfilled for a successful installation:

- Download and extract a zip of **SRSMP** from the `GitHub page <https://github.com/NixtonM/srsmp>`_.
- Install **Thorlabs OSA** drivers (included within the *repository*)
- Install a **C compiler** (either via *Microsoft Visual Studio* or *MinGW*)
- Install **Python** (implementation written and tested on *v3.6*)

We also recommend setting up a new *virtual environment* within *anaconda*:

.. code-block:: shell

   $ conda create --name srsmp_env python=3.6


Package Installation
-------------------------
All dependencies for a functional installation have been defined within `setup.py` and therefore can be 
installed by:

.. code-block:: shell

   $ cd /path/to/repo
   $ pip install .
	

**SRSMP** uses `Instrumental <https://instrumental-lib.readthedocs.io/>`_ by *Nate Bogdanowicz*. At the time 
of the deployment of **SRSMP**, required features within *Instrumental* were only available in the current 
developer version. Therefore a wheel package of *Instrumental* (v0.6dev0) is provided within the repository. 
To install perform:

.. code-block:: shell

   $ cd /path/to/repo/libs/
   $ pip install Instrumental_lib-0.6.dev0-cp36-cp36m-win_amd64.whl

