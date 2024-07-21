------------
Installation
------------

Using `pip`
===========

You can use `pip` to install **mpcrl** with the command

.. code:: bash

   pip install mpcrl

**mpcrl** has the following dependencies

-  Python 3.9 or higher
- `csnlp <https://casadi-nlp.readthedocs.io/en/stable/>`__
- `SciPy <https://scipy.org/>`__
- `Gymnasium <https://gymnasium.farama.org/>`__
- `Numba <https://numba.pydata.org/>`__
- `typing_extensions <https://pypi.org/project/typing-extensions/>`__ (only for Python
  3.9)

Using source code
=================

If you'd like to play around with the source code instead, run

.. code:: bash

   git clone https://github.com/FilippoAiraldi/mpc-reinforcement-learning.git

The `main` branch contains the main releases of the packages (and the occasional post
release). The `experimental` branch is reserved for the implementation and test of new
features and hosts the release candidates. You can then install the package to edit it
as you wish as

.. code:: bash

   pip install -e /path/to/mpc-reinforcement-learning
