from glob import glob

from setuptools import find_packages, setup


package_name = "s622_dual_assembly"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dsl",
    maintainer_email="1197226282@qq.com",
    description="Force-guided planar spiral search for S622 assembly.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "control_mode = s622_dual_assembly.control_mode:main",
            "planar_search_node = s622_dual_assembly.planar_search_node:main",
        ],
    },
)
