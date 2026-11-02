# Third-party code and model

The bundled CALVADOS code is from the
[KULL-Centre/CALVADOS repository](https://github.com/KULL-Centre/CALVADOS),
revision `dc5beac158b72262f585947487c1dd8ef1602a18`.
Its license is retained in [CALVADOS_LICENSE](simulation/vendor/CALVADOS_LICENSE).

This copy retains system construction, force definitions, and selected numerical
helpers. The upstream simulation runner, plotting, and blocking-analysis utilities
are omitted; configuration and package imports are adapted to this subset.

The two-bead DNA model follows Kapoor, Kim, and Mittal, J. Chem. Theory Comput.
(2024), [DOI: 10.1021/acs.jctc.3c00525](https://doi.org/10.1021/acs.jctc.3c00525).
The OpenMM implementation is in `simulation/common/kapoor2b_openmm.py`.
