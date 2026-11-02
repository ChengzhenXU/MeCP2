"""Portable OpenMM simulation construction."""


def create_simulation(topology, system, integrator, platform_name=None):
    from openmm import Platform, app

    if platform_name is None:
        for index in range(Platform.getNumPlatforms()):
            platform = Platform.getPlatform(index)
            if "Precision" in platform.getPropertyNames():
                platform.setPropertyDefaultValue("Precision", "mixed")
        return app.Simulation(topology, system, integrator)

    platform = Platform.getPlatformByName(platform_name)
    properties = {"Precision": "mixed"} if "Precision" in platform.getPropertyNames() else {}
    return app.Simulation(topology, system, integrator, platform, properties)
